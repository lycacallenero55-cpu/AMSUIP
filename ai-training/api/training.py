from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import os
import uuid
from datetime import datetime
import time
import logging
import asyncio

from models.database import db_manager
from models.signature_model import SignatureVerificationModel
from utils.image_processing import preprocess_image
from utils.storage import save_to_supabase, cleanup_local_file
from utils.s3_storage import upload_model_file
from utils.augmentation import SignatureAugmentation
from utils.job_queue import job_queue
from utils.training_callback import RealTimeMetricsCallback
from services.model_versioning import model_versioning_service
from config import settings
from models.global_signature_model import GlobalSignatureVerificationModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance - can handle up to 150 students
signature_ai_manager = SignatureVerificationModel(max_students=150)

async def train_signature_model(student, genuine_data, forged_data, job=None):
    try:
        # Process and validate images
        genuine_images = []
        forged_images = []

        for i, data in enumerate(genuine_data):
            image = Image.open(io.BytesIO(data))
            genuine_images.append(image)
            if job:
                progress = 5.0 + (i + 1) / len(genuine_data) * 15.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing genuine images... {i+1}/{len(genuine_data)}")

        for i, data in enumerate(forged_data):
            image = Image.open(io.BytesIO(data))
            forged_images.append(image)
            if job:
                progress = 20.0 + (i + 1) / len(forged_data) * 15.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing forged images... {i+1}/{len(forged_data)}")

        if job:
            job_queue.update_job_progress(job.job_id, 35.0, "Preparing training data...")

        training_data = {
            f"student_{student['id']}": {
                'genuine': genuine_images,
                'forged': forged_images
            }
        }

        if job:
            job_queue.update_job_progress(job.job_id, 50.0, "Training student and authenticity models...")

        t0 = time.time()
        result_models = signature_ai_manager.train_system(training_data)

        if job:
            job_queue.update_job_progress(job.job_id, 80.0, "Saving models...")

        model_uuid = str(uuid.uuid4())
        base_path = os.path.join(settings.LOCAL_MODELS_DIR, f"signature_model_{model_uuid}")
        os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)

        signature_ai_manager.save_models(base_path)

        # Upload models to S3 instead of Supabase
        with open(f"{base_path}_student_model.keras", 'rb') as f:
            student_model_data = f.read()
        with open(f"{base_path}_authenticity_model.keras", 'rb') as f:
            auth_model_data = f.read()
        
        # Upload to S3 with organized folder structure
        student_s3_key, student_s3_url = upload_model_file(
            student_model_data, "individual", f"student_{model_uuid}", "keras"
        )
        auth_s3_key, auth_s3_url = upload_model_file(
            auth_model_data, "individual", f"auth_{model_uuid}", "keras"
        )

        cleanup_local_file(f"{base_path}_student_model.keras")
        cleanup_local_file(f"{base_path}_authenticity_model.keras")
        cleanup_local_file(f"{base_path}_student_mappings.json")

        model_record = await db_manager.create_trained_model({
            "student_id": int(student["id"]),
            "model_path": student_s3_url,
            "embedding_model_path": auth_s3_url,
            "status": "completed",
            "sample_count": len(genuine_images) + len(forged_images),
            "genuine_count": len(genuine_images),
            "forged_count": len(forged_images),
            "training_date": datetime.utcnow().isoformat(),
            "training_metrics": {
                'model_type': 'individual_recognition',
                'student_recognition_accuracy': float(result_models['student_history'].get('accuracy', [0])[-1]) if 'student_history' in result_models else None,
                'authenticity_accuracy': float(result_models['authenticity_history'].get('accuracy', [0])[-1]) if 'authenticity_history' in result_models else None,
                'epochs_trained': len(result_models['student_history'].get('accuracy', [])) if 'student_history' in result_models else None
            }
        })

        train_time = time.time() - t0
        result = {
            "success": True,
            "model_id": model_record.get("id") if isinstance(model_record, dict) else None,
            "model_uuid": model_uuid,
            "train_time_s": float(train_time),
            "training_samples": len(genuine_images) + len(forged_images),
            "genuine_count": len(genuine_images),
            "forged_count": len(forged_images)
        }

        if job:
            job_queue.complete_job(job.job_id, result)
        return result
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if job:
            job_queue.fail_job(job.job_id, str(e))
        raise

@router.post("/start")
async def start_training(
    student_id: str = Form(...),
    genuine_files: List[UploadFile] = File(...),
    forged_files: List[UploadFile] = File(...)
):
    try:
        student = await db_manager.get_student_by_school_id(student_id)
        if not student:
            try:
                numeric_id = int(student_id)
                student = await db_manager.get_student(numeric_id)
            except Exception:
                student = None
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
            raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
        if len(forged_files) < settings.MIN_FORGED_SAMPLES:
            raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_FORGED_SAMPLES} forged samples required")

        genuine_data = [await f.read() for f in genuine_files]
        forged_data = [await f.read() for f in forged_files]

        result = await train_signature_model(student, genuine_data, forged_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/start-async")
async def start_async_training(
    student_id: str = Form(...),
    genuine_files: List[UploadFile] = File(...),
    forged_files: List[UploadFile] = File(...)
):
    try:
        # Handle multiple students (comma-separated) or single student
        student_ids = [sid.strip() for sid in student_id.split(',') if sid.strip()]
        
        if len(student_ids) == 1:
            # Single student training (original logic)
            student = await db_manager.get_student_by_school_id(student_ids[0])
            if not student:
                try:
                    numeric_id = int(student_ids[0])
                    student = await db_manager.get_student(numeric_id)
                except Exception:
                    student = None
            if not student:
                raise HTTPException(status_code=404, detail="Student not found")

            if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
                raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
            if len(forged_files) < settings.MIN_FORGED_SAMPLES:
                raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_FORGED_SAMPLES} forged samples required")

            job = job_queue.create_job(int(student["id"]), "training")
            genuine_data = [await f.read() for f in genuine_files]
            forged_data = [await f.read() for f in forged_files]
            asyncio.create_task(run_async_training(job, student, genuine_data, forged_data))
            return {"success": True, "job_id": job.job_id, "message": "Training job started", "stream_url": f"/api/progress/stream/{job.job_id}"}
        
        else:
            # Multiple students - use global training
            if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
                raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
            if len(forged_files) < settings.MIN_FORGED_SAMPLES:
                raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_FORGED_SAMPLES} forged samples required")

            # Create a job for global training
            job = job_queue.create_job(0, "global_training")  # 0 indicates global training
            genuine_data = [await f.read() for f in genuine_files]
            forged_data = [await f.read() for f in forged_files]
            asyncio.create_task(run_global_async_training(job, student_ids, genuine_data, forged_data))
            return {"success": True, "job_id": job.job_id, "message": "Global training job started", "stream_url": f"/api/progress/stream/{job.job_id}"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting async training: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/train-global")
async def train_global_model():
    try:
        # Build manifest from DB (S3 URLs)
        rows = await db_manager.list_all_signatures()
        if not rows:
            raise HTTPException(status_code=400, detail="No signatures available")

        import requests, io
        from PIL import Image
        from utils.image_processing import preprocess_image

        data_by_student = {}
        for r in rows:
            sid = int(r["student_id"])  # type: ignore[index]
            url = r["s3_url"]  # type: ignore[index]
            label = r["label"]  # type: ignore[index]
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
            image = preprocess_image(image)
            bucket = data_by_student.setdefault(sid, {"genuine_images": [], "forged_images": []})
            (bucket["genuine_images"] if label == "genuine" else bucket["forged_images"]).append(image)

        gsm = GlobalSignatureVerificationModel()
        history = gsm.train_global_model(data_by_student)
        
        # Save global model to S3
        model_uuid = str(uuid.uuid4())
        base_path = os.path.join(settings.LOCAL_MODELS_DIR, f"global_model_{model_uuid}")
        os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)
        
        # Save model locally first
        gsm.save_model(f"{base_path}.keras")
        
        # Upload to S3
        with open(f"{base_path}.keras", 'rb') as f:
            model_data = f.read()
        
        s3_key, s3_url = upload_model_file(
            model_data, "global", f"global_{model_uuid}", "keras"
        )
        
        # Clean up local file
        cleanup_local_file(f"{base_path}.keras")
        
        # Store global model record in dedicated global table
        model_record = await db_manager.create_global_model({
            "model_path": s3_url,
            "s3_key": s3_key,
            "model_uuid": model_uuid,
            "status": "completed",
            "sample_count": sum(len(data['genuine_images']) + len(data['forged_images']) for data in data_by_student.values()),
            "genuine_count": sum(len(data['genuine_images']) for data in data_by_student.values()),
            "forged_count": sum(len(data['forged_images']) for data in data_by_student.values()),
            "student_count": len(data_by_student),
            "training_date": datetime.utcnow().isoformat(),
            "accuracy": float(history.history.get('accuracy', [0])[-1]) if history.history.get('accuracy') else None,
            "training_metrics": {
                'model_type': 'global_multi_student',
                'final_accuracy': float(history.history.get('accuracy', [0])[-1]) if history.history.get('accuracy') else None,
                'final_loss': float(history.history.get('loss', [0])[-1]) if history.history.get('loss') else None,
                'epochs_trained': len(history.history.get('accuracy', [])),
                'val_accuracy': float(history.history.get('val_accuracy', [0])[-1]) if history.history.get('val_accuracy') else None,
                'val_loss': float(history.history.get('val_loss', [0])[-1]) if history.history.get('val_loss') else None
            }
        })
        
        return {
            "success": True, 
            "history": {k: list(map(float, v)) for k, v in history.history.items()},
            "model_id": model_record.get("id") if isinstance(model_record, dict) else None,
            "model_uuid": model_uuid,
            "s3_url": s3_url
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Global training failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def run_async_training(job, student, genuine_data, forged_data):
    try:
        job_queue.start_job(job.job_id)
        await train_signature_model(student, genuine_data, forged_data, job)
    except Exception as e:
        logger.error(f"Async training failed: {e}")
        job_queue.fail_job(job.job_id, str(e))


async def run_global_async_training(job, student_ids, genuine_data, forged_data):
    """Run global training for multiple students using uploaded files"""
    try:
        job_queue.start_job(job.job_id)
        job_queue.update_job_progress(job.job_id, 10.0, "Processing uploaded files...")
        
        # Group files by student (this is a simplified approach)
        # In a real implementation, you'd need to know which files belong to which student
        # For now, we'll distribute files evenly among students
        
        students = []
        for student_id in student_ids:
            student = await db_manager.get_student_by_school_id(student_id)
            if not student:
                try:
                    numeric_id = int(student_id)
                    student = await db_manager.get_student(numeric_id)
                except Exception:
                    continue
            if student:
                students.append(student)
        
        if not students:
            raise Exception("No valid students found")
        
        job_queue.update_job_progress(job.job_id, 20.0, f"Found {len(students)} students, processing images...")
        
        # Process images
        genuine_images = []
        forged_images = []
        
        for i, data in enumerate(genuine_data):
            image = Image.open(io.BytesIO(data))
            genuine_images.append(image)
            if job:
                progress = 20.0 + (i + 1) / len(genuine_data) * 30.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing genuine images... {i+1}/{len(genuine_data)}")

        for i, data in enumerate(forged_data):
            image = Image.open(io.BytesIO(data))
            forged_images.append(image)
            if job:
                progress = 50.0 + (i + 1) / len(forged_data) * 20.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing forged images... {i+1}/{len(forged_data)}")

        if job:
            job_queue.update_job_progress(job.job_id, 70.0, "Preparing training data...")

        # Create training data structure for global model
        # Distribute images evenly among students (simplified approach)
        images_per_student = len(genuine_images) // len(students)
        forged_per_student = len(forged_images) // len(students)
        
        training_data = {}
        for i, student in enumerate(students):
            start_idx = i * images_per_student
            end_idx = start_idx + images_per_student if i < len(students) - 1 else len(genuine_images)
            
            forged_start = i * forged_per_student
            forged_end = forged_start + forged_per_student if i < len(students) - 1 else len(forged_images)
            
            training_data[f"student_{student['id']}"] = {
                'genuine_images': genuine_images[start_idx:end_idx],
                'forged_images': forged_images[forged_start:forged_end]
            }

        if job:
            job_queue.update_job_progress(job.job_id, 80.0, "Training global model...")

        # Train global model
        gsm = GlobalSignatureVerificationModel()
        history = gsm.train_global_model(training_data)
        
        # Save global model to S3
        model_uuid = str(uuid.uuid4())
        base_path = os.path.join(settings.LOCAL_MODELS_DIR, f"global_model_{model_uuid}")
        os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)
        
        gsm.save_model(f"{base_path}.keras")
        
        with open(f"{base_path}.keras", 'rb') as f:
            model_data = f.read()
        
        s3_key, s3_url = upload_model_file(
            model_data, "global", f"global_{model_uuid}", "keras"
        )
        
        cleanup_local_file(f"{base_path}.keras")
        
        # Store global model record in dedicated global table
        model_record = await db_manager.create_global_model({
            "model_path": s3_url,
            "s3_key": s3_key,
            "model_uuid": model_uuid,
            "status": "completed",
            "sample_count": len(genuine_images) + len(forged_images),
            "genuine_count": len(genuine_images),
            "forged_count": len(forged_images),
            "student_count": len(students),
            "training_date": datetime.utcnow().isoformat(),
            "accuracy": float(history.history.get('accuracy', [0])[-1]) if history.history.get('accuracy') else None,
            "training_metrics": {
                'model_type': 'global_multi_student',
                'final_accuracy': float(history.history.get('accuracy', [0])[-1]) if history.history.get('accuracy') else None,
                'final_loss': float(history.history.get('loss', [0])[-1]) if history.history.get('loss') else None,
                'epochs_trained': len(history.history.get('accuracy', [])),
                'val_accuracy': float(history.history.get('val_accuracy', [0])[-1]) if history.history.get('val_accuracy') else None,
                'val_loss': float(history.history.get('val_loss', [0])[-1]) if history.history.get('val_loss') else None
            }
        })
        
        result = {
            "success": True,
            "model_id": model_record.get("id") if isinstance(model_record, dict) else None,
            "model_uuid": model_uuid,
            "s3_url": s3_url,
            "student_count": len(students),
            "training_samples": len(genuine_images) + len(forged_images)
        }
        
        if job:
            job_queue.complete_job(job.job_id, result)
            
    except Exception as e:
        logger.error(f"Global async training failed: {e}")
        if job:
            job_queue.fail_job(job.job_id, str(e))

@router.get("/models")
async def get_trained_models(student_id: Optional[int] = None):
    try:
        models = await db_manager.get_trained_models(student_id)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting trained models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/global-models")
async def get_global_models(limit: Optional[int] = None):
    try:
        models = await db_manager.get_global_models(limit)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting global models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/global-models/latest")
async def get_latest_global_model():
    try:
        model = await db_manager.get_latest_global_model()
        if not model:
            raise HTTPException(status_code=404, detail="No global models found")
        return {"model": model}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest global model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
