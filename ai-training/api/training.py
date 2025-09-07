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

        remote_student_path = await save_to_supabase(f"{base_path}_student_model.keras", f"models/signature_student_model_{model_uuid}.keras")
        remote_auth_path = await save_to_supabase(f"{base_path}_authenticity_model.keras", f"models/signature_authenticity_model_{model_uuid}.keras")

        cleanup_local_file(f"{base_path}_student_model.keras")
        cleanup_local_file(f"{base_path}_authenticity_model.keras")
        cleanup_local_file(f"{base_path}_student_mappings.json")

        model_record = await db_manager.create_trained_model({
            "student_id": int(student["id"]),
            "model_path": remote_student_path,
            "embedding_model_path": remote_auth_path,
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

        job = job_queue.create_job(int(student["id"]), "training")
        genuine_data = [await f.read() for f in genuine_files]
        forged_data = [await f.read() for f in forged_files]
        asyncio.create_task(run_async_training(job, student, genuine_data, forged_data))
        return {"success": True, "job_id": job.job_id, "message": "Training job started", "stream_url": f"/api/progress/stream/{job.job_id}"}
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
        return {"success": True, "history": {k: list(map(float, v)) for k, v in history.history.items()}}
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

@router.get("/models")
async def get_trained_models(student_id: Optional[int] = None):
    try:
        models = await db_manager.get_trained_models(student_id)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting trained models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")