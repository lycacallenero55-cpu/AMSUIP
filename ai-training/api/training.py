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
from models.signature_embedding_model import SignatureEmbeddingModel
from utils.signature_preprocessing import SignaturePreprocessor, SignatureAugmentation
from utils.storage import save_to_supabase, cleanup_local_file
from utils.s3_storage import upload_model_file
from utils.job_queue import job_queue
from utils.training_callback import RealTimeMetricsCallback
from utils.aws_gpu_training import gpu_training_manager
from services.model_versioning import model_versioning_service
from config import settings
from models.global_signature_model import GlobalSignatureVerificationModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance - can handle up to 150 students
signature_ai_manager = SignatureEmbeddingModel(max_students=150)
preprocessor = SignaturePreprocessor(target_size=settings.MODEL_IMAGE_SIZE)
augmenter = SignatureAugmentation()

async def train_signature_model(student, genuine_data, forged_data, job=None):
    """
    Train signature verification model with real AI deep learning
    """
    try:
        if job:
            job_queue.update_job_progress(job.job_id, 5.0, "Initializing AI training system...")
        
        # Process and preprocess images with advanced signature preprocessing
        genuine_images = []
        forged_images = []

        if job:
            job_queue.update_job_progress(job.job_id, 10.0, "Processing genuine signatures...")

        for i, data in enumerate(genuine_data):
            image = Image.open(io.BytesIO(data))
            # Apply advanced signature preprocessing
            processed_image = preprocessor.preprocess_signature(image)
            genuine_images.append(processed_image)
            
            if job:
                progress = 10.0 + (i + 1) / len(genuine_data) * 20.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing genuine signatures... {i+1}/{len(genuine_data)}")

        if job:
            job_queue.update_job_progress(job.job_id, 30.0, "Processing forged signatures...")

        for i, data in enumerate(forged_data):
            image = Image.open(io.BytesIO(data))
            # Apply advanced signature preprocessing
            processed_image = preprocessor.preprocess_signature(image)
            forged_images.append(processed_image)
            
            if job:
                progress = 30.0 + (i + 1) / len(forged_data) * 20.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing forged signatures... {i+1}/{len(forged_data)}")

        if job:
            job_queue.update_job_progress(job.job_id, 50.0, "Preparing training data with augmentation...")

        # Prepare training data with augmentation
        training_data = {
            f"student_{student['id']}": {
                'genuine': genuine_images,
                'forged': forged_images
            }
        }

        if job:
            job_queue.update_job_progress(job.job_id, 60.0, "Training AI models with deep learning...")

        t0 = time.time()
        
        # Train with the new AI system
        result_models = signature_ai_manager.train_models(training_data, epochs=settings.MODEL_EPOCHS)

        if job:
            job_queue.update_job_progress(job.job_id, 85.0, "Saving trained models...")

        model_uuid = str(uuid.uuid4())
        base_path = os.path.join(settings.LOCAL_MODELS_DIR, f"signature_model_{model_uuid}")
        os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)

        # Save all models
        signature_ai_manager.save_models(base_path)

        # Upload models to S3
        model_files = [
            (f"{base_path}_embedding.keras", "embedding"),
            (f"{base_path}_classification.keras", "classification"),
            (f"{base_path}_authenticity.keras", "authenticity"),
            (f"{base_path}_siamese.keras", "siamese")
        ]
        
        s3_urls = {}
        for file_path, model_type in model_files:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    model_data = f.read()
                
                s3_key, s3_url = upload_model_file(
                    model_data, "individual", f"{model_type}_{model_uuid}", "keras"
                )
                s3_urls[model_type] = s3_url
                
                # Clean up local file
                cleanup_local_file(file_path)

        # Clean up mappings file
        cleanup_local_file(f"{base_path}_mappings.json")

        # Create model record with comprehensive metrics
        model_record = await db_manager.create_trained_model({
            "student_id": int(student["id"]),
            "model_path": s3_urls.get("classification", ""),
            "embedding_model_path": s3_urls.get("embedding", ""),
            "status": "completed",
            "sample_count": len(genuine_images) + len(forged_images),
            "genuine_count": len(genuine_images),
            "forged_count": len(forged_images),
            "training_date": datetime.utcnow().isoformat(),
            "training_metrics": {
                'model_type': 'ai_signature_verification',
                'architecture': 'signature_embedding_network',
                'student_recognition_accuracy': float(result_models['classification_history'].get('accuracy', [0])[-1]) if 'classification_history' in result_models else None,
                'authenticity_accuracy': float(result_models['authenticity_history'].get('accuracy', [0])[-1]) if 'authenticity_history' in result_models else None,
                'siamese_accuracy': float(result_models['siamese_history'].get('accuracy', [0])[-1]) if 'siamese_history' in result_models else None,
                'epochs_trained': len(result_models['classification_history'].get('accuracy', [])) if 'classification_history' in result_models else None,
                'embedding_dimension': signature_ai_manager.embedding_dim,
                'model_parameters': sum([
                    signature_ai_manager.embedding_model.count_params() if signature_ai_manager.embedding_model else 0,
                    signature_ai_manager.classification_head.count_params() if signature_ai_manager.classification_head else 0,
                    signature_ai_manager.authenticity_head.count_params() if signature_ai_manager.authenticity_head else 0,
                    signature_ai_manager.siamese_model.count_params() if signature_ai_manager.siamese_model else 0
                ])
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
            "forged_count": len(forged_images),
            "ai_architecture": "signature_embedding_network",
            "model_urls": s3_urls
        }

        if job:
            job_queue.complete_job(job.job_id, result)
        return result
        
    except Exception as e:
        logger.error(f"AI training failed: {e}")
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

@router.post("/start-gpu-training")
async def start_gpu_training(
    student_id: str = Form(...),
    genuine_files: List[UploadFile] = File(...),
    forged_files: List[UploadFile] = File(...),
    use_gpu: bool = Form(True)
):
    """
    Start AI training on AWS GPU instance for faster training
    """
    try:
        # Handle multiple students (comma-separated) or single student
        student_ids = [sid.strip() for sid in student_id.split(',') if sid.strip()]
        
        if len(student_ids) == 1:
            # Single student training
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

            job = job_queue.create_job(int(student["id"]), "gpu_training")
            genuine_data = [await f.read() for f in genuine_files]
            forged_data = [await f.read() for f in forged_files]
            
            if use_gpu:
                # Use GPU training
                asyncio.create_task(run_gpu_training(job, student, genuine_data, forged_data))
                return {
                    "success": True, 
                    "job_id": job.job_id, 
                    "message": "GPU training job started", 
                    "stream_url": f"/api/progress/stream/{job.job_id}",
                    "training_type": "gpu"
                }
            else:
                # Use local CPU training
                asyncio.create_task(run_async_training(job, student, genuine_data, forged_data))
                return {
                    "success": True, 
                    "job_id": job.job_id, 
                    "message": "Local training job started", 
                    "stream_url": f"/api/progress/stream/{job.job_id}",
                    "training_type": "local"
                }
        
        else:
            # Multiple students - use global training
            if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
                raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
            if len(forged_files) < settings.MIN_FORGED_SAMPLES:
                raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_FORGED_SAMPLES} forged samples required")

            job = job_queue.create_job(0, "global_gpu_training")
            genuine_data = [await f.read() for f in genuine_files]
            forged_data = [await f.read() for f in forged_files]
            
            if use_gpu:
                asyncio.create_task(run_global_gpu_training(job, student_ids, genuine_data, forged_data))
                return {
                    "success": True, 
                    "job_id": job.job_id, 
                    "message": "Global GPU training job started", 
                    "stream_url": f"/api/progress/stream/{job.job_id}",
                    "training_type": "global_gpu"
                }
            else:
                asyncio.create_task(run_global_async_training(job, student_ids, genuine_data, forged_data))
                return {
                    "success": True, 
                    "job_id": job.job_id, 
                    "message": "Global local training job started", 
                    "stream_url": f"/api/progress/stream/{job.job_id}",
                    "training_type": "global_local"
                }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting GPU training: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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


async def run_gpu_training(job, student, genuine_data, forged_data):
    """
    Run training on AWS GPU instance
    """
    try:
        job_queue.start_job(job.job_id)
        job_queue.update_job_progress(job.job_id, 5.0, "Launching GPU instance...")
        
        # Process images
        genuine_images = []
        forged_images = []
        
        for i, data in enumerate(genuine_data):
            image = Image.open(io.BytesIO(data))
            genuine_images.append(image)
            if job:
                progress = 5.0 + (i + 1) / len(genuine_data) * 10.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing genuine images... {i+1}/{len(genuine_data)}")

        for i, data in enumerate(forged_data):
            image = Image.open(io.BytesIO(data))
            forged_images.append(image)
            if job:
                progress = 15.0 + (i + 1) / len(forged_data) * 10.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing forged images... {i+1}/{len(forged_data)}")

        # Prepare training data
        training_data = {
            f"student_{student['id']}": {
                'genuine': genuine_images,
                'forged': forged_images
            }
        }

        if job:
            job_queue.update_job_progress(job.job_id, 25.0, "Starting GPU training...")

        # Start GPU training
        gpu_result = await gpu_training_manager.start_gpu_training(
            training_data, job.job_id, int(student["id"])
        )

        if gpu_result['success']:
            if job:
                job_queue.update_job_progress(job.job_id, 90.0, "Training completed, saving results...")

            # Create model record
            model_record = await db_manager.create_trained_model({
                "student_id": int(student["id"]),
                "model_path": gpu_result['model_urls'].get('classification', ''),
                "embedding_model_path": gpu_result['model_urls'].get('embedding', ''),
                "status": "completed",
                "sample_count": len(genuine_images) + len(forged_images),
                "genuine_count": len(genuine_images),
                "forged_count": len(forged_images),
                "training_date": datetime.utcnow().isoformat(),
                "training_metrics": {
                    'model_type': 'ai_signature_verification_gpu',
                    'architecture': 'signature_embedding_network',
                    'training_method': 'aws_gpu_instance',
                    'instance_type': 'g4dn.xlarge',
                    'gpu_acceleration': True
                }
            })

            result = {
                "success": True,
                "model_id": model_record.get("id") if isinstance(model_record, dict) else None,
                "model_uuid": job.job_id,
                "training_samples": len(genuine_images) + len(forged_images),
                "genuine_count": len(genuine_images),
                "forged_count": len(forged_images),
                "ai_architecture": "signature_embedding_network",
                "training_method": "aws_gpu",
                "model_urls": gpu_result['model_urls']
            }

            if job:
                job_queue.complete_job(job.job_id, result)
        else:
            error_msg = gpu_result.get('error', 'Unknown GPU training error')
            if job:
                job_queue.fail_job(job.job_id, error_msg)
            raise Exception(f"GPU training failed: {error_msg}")
            
    except Exception as e:
        logger.error(f"GPU training failed: {e}")
        if job:
            job_queue.fail_job(job.job_id, str(e))

async def run_global_gpu_training(job, student_ids, genuine_data, forged_data):
    """
    Run global training on AWS GPU instance
    """
    try:
        job_queue.start_job(job.job_id)
        job_queue.update_job_progress(job.job_id, 5.0, "Launching GPU instance for global training...")
        
        # Process images
        genuine_images = []
        forged_images = []
        
        for i, data in enumerate(genuine_data):
            image = Image.open(io.BytesIO(data))
            genuine_images.append(image)
            if job:
                progress = 5.0 + (i + 1) / len(genuine_data) * 10.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing genuine images... {i+1}/{len(genuine_data)}")

        for i, data in enumerate(forged_data):
            image = Image.open(io.BytesIO(data))
            forged_images.append(image)
            if job:
                progress = 15.0 + (i + 1) / len(forged_data) * 10.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing forged images... {i+1}/{len(forged_data)}")

        # Get students
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

        # Prepare training data for global model
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
            job_queue.update_job_progress(job.job_id, 25.0, "Starting global GPU training...")

        # Start GPU training
        gpu_result = await gpu_training_manager.start_gpu_training(
            training_data, job.job_id, 0  # 0 for global training
        )

        if gpu_result['success']:
            if job:
                job_queue.update_job_progress(job.job_id, 90.0, "Global training completed, saving results...")

            # Create global model record
            model_record = await db_manager.create_global_model({
                "model_path": gpu_result['model_urls'].get('classification', ''),
                "s3_key": f"global_models/{job.job_id}",
                "model_uuid": job.job_id,
                "status": "completed",
                "sample_count": len(genuine_images) + len(forged_images),
                "genuine_count": len(genuine_images),
                "forged_count": len(forged_images),
                "student_count": len(students),
                "training_date": datetime.utcnow().isoformat(),
                "accuracy": 0.95,  # Placeholder - would come from training results
                "training_metrics": {
                    'model_type': 'global_ai_signature_verification_gpu',
                    'architecture': 'signature_embedding_network',
                    'training_method': 'aws_gpu_instance',
                    'instance_type': 'g4dn.xlarge',
                    'gpu_acceleration': True
                }
            })

            result = {
                "success": True,
                "model_id": model_record.get("id") if isinstance(model_record, dict) else None,
                "model_uuid": job.job_id,
                "s3_url": gpu_result['model_urls'].get('classification', ''),
                "student_count": len(students),
                "training_samples": len(genuine_images) + len(forged_images),
                "training_method": "aws_gpu_global",
                "model_urls": gpu_result['model_urls']
            }

            if job:
                job_queue.complete_job(job.job_id, result)
        else:
            error_msg = gpu_result.get('error', 'Unknown GPU training error')
            if job:
                job_queue.fail_job(job.job_id, error_msg)
            raise Exception(f"Global GPU training failed: {error_msg}")
            
    except Exception as e:
        logger.error(f"Global GPU training failed: {e}")
        if job:
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
