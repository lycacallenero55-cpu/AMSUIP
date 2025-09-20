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
from tensorflow import keras

try:
    from models.database import db_manager
except Exception as e:
    print(f"Warning: Database manager not available: {e}")
    db_manager = None

def check_database_available():
    """Check if database is available for operations"""
    if db_manager is None or db_manager.client is None:
        raise HTTPException(status_code=503, detail="Database not available - running in offline mode")
from models.signature_embedding_model import SignatureEmbeddingModel
from utils.signature_preprocessing import SignaturePreprocessor, SignatureAugmentation
# Removed unused Supabase imports - using S3 directly
from utils.s3_storage import upload_model_file
from utils.direct_s3_saving import save_signature_models_directly, DirectS3ModelSaver
from utils.optimized_s3_saving import save_signature_models_optimized
from utils.job_queue import job_queue
from utils.training_callback import RealTimeMetricsCallback
from utils.aws_gpu_training import gpu_training_manager
from services.model_versioning import model_versioning_service
from config import settings
import os
from models.global_signature_model import GlobalSignatureVerificationModel
from utils.s3_storage import create_presigned_get, download_bytes
from utils.s3_storage import upload_model_file as upload_file_generic
from utils.storage import cleanup_local_file
from utils.artifacts import package_global_classifier_artifacts, build_classifier_spec

router = APIRouter()
logger = logging.getLogger(__name__)

# Global instances removed - each training session creates its own instances
# This prevents state contamination between different training requests

@router.get("/gpu-available")
async def gpu_available():
    try:
        available = gpu_training_manager.is_available()
        return {"available": bool(available)}
    except Exception:
        return {"available": False}

def _derive_s3_key_from_url(url: str) -> str | None:
    """Best-effort derive S3 key from a public-style URL.

    Examples:
      https://bucket.s3.us-east-1.amazonaws.com/path/to/object -> path/to/object
      https://s3.us-east-1.amazonaws.com/bucket/path/to/object -> bucket/path/to/object (less common)
    """
    if not url:
        return None
    # Strip query string if present
    base = url.split('?', 1)[0]
    if "amazonaws.com" not in base:
        return None
    try:
        # Most common style: https://{bucket}.s3.{region}.amazonaws.com/{key}
        parts = base.split(".amazonaws.com/")
        if len(parts) == 2:
            return parts[1] or None
        # Fallback: split first occurrence of domain path
        return base.split("/", 3)[-1] or None
    except Exception:
        return None

async def _fetch_and_validate_student_images(student_ids: list[int]) -> dict[int, dict[str, list]]:
    """Fetch signature images from DB/S3 for the given students and return preprocessed arrays.

    Returns mapping: { student_id: { 'genuine_images': [np.ndarray], 'forged_images': [np.ndarray] } }
    Invalid or non-image URLs are skipped.
    """
    import requests
    from PIL import Image
    import io

    results: dict[int, dict[str, list]] = {}
    
    # Create fresh preprocessor instance for this function
    preprocessor = SignaturePreprocessor(target_size=settings.MODEL_IMAGE_SIZE)

    import asyncio as _asyncio
    import functools as _functools

    async def _process_student(sid: int):
        rows = await db_manager.list_student_signatures(int(sid))
        bucket = {"genuine_images": [], "forged_images": []}
        accepted_g = 0
        accepted_f = 0
        skipped = 0

        async def _fetch_one(r):
            nonlocal accepted_g, accepted_f, skipped
            url = r.get("s3_url")
            label = (r.get("label") or "").lower()
            if not url:
                return
            try:
                content: bytes | None = None
                key = r.get("s3_key") or _derive_s3_key_from_url(url)
                if key:
                    try:
                        content = download_bytes(key)
                    except Exception as e:
                        logger.debug(f"S3 download failed for key={key}: {e}")
                        content = None
                if content is None:
                    if settings.S3_USE_PRESIGNED_GET and key:
                        try:
                            url = create_presigned_get(key)
                        except Exception:
                            pass
                    resp = requests.get(url, timeout=6)
                    if resp.status_code != 200:
                        skipped += 1
                        return
                    content = resp.content
                bio = io.BytesIO(content)
                img = Image.open(bio).convert('RGB')
                arr = preprocessor.preprocess_signature(img)
                if label == "genuine":
                    bucket["genuine_images"].append(arr)
                    accepted_g += 1
                else:
                    bucket["forged_images"].append(arr)
                    accepted_f += 1
            except Exception:
                skipped += 1
                return

        # bounded parallelism per student
        sem = _asyncio.Semaphore(6)
        async def _sem_task(r):
            async with sem:
                await _fetch_one(r)

        await _asyncio.gather(*[_sem_task(r) for r in (rows or [])])
        logger.info(f"Student {sid}: accepted {accepted_g} genuine, {accepted_f} forged, skipped {skipped}")
        results[int(sid)] = bucket

    # Run students sequentially to limit total concurrency; per-student work is parallelized
    for sid in student_ids:
        await _process_student(int(sid))
    return results


async def _train_and_store_individual_from_arrays(student: dict, genuine_arrays: list, forged_arrays: list, job=None, global_model_id: int | None = None, use_s3_upload: bool = False) -> dict:
    """Train and store an individual model given preprocessed arrays (hybrid mode helper)."""
    if job:
        job_queue.update_job_progress(job.job_id, 92.0, f"Training individual model for student {student['id']} ({len(genuine_arrays)}G/{len(forged_arrays)}F)...")
    
    # Normalize arrays to float32 [H,W,3]
    import numpy as np
    def _normalize(img):
        # Ensure it's a numpy array first
        if not isinstance(img, np.ndarray):
            arr = np.array(img)
        else:
            arr = img
        
        # Convert to float32 if needed
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        
        # Handle batched arrays
        if arr.ndim == 4:
            # If erroneously batched, squeeze first dim when size 1
            if arr.shape[0] == 1:
                arr = arr[0]
        return arr
    genuine_norm = [_normalize(a) for a in genuine_arrays]
    forged_norm = [_normalize(a) for a in forged_arrays]
    
    # Use student name for consistent mapping
    student_name = f"{student.get('firstname', '')} {student.get('surname', '')}".strip() or f"Student_{student['id']}"
    training_data = {
        student_name: {
            'genuine': genuine_norm,
            'forged': forged_norm
        }
    }

    # Use a fresh model manager per student to avoid cross-contamination across sequential trainings
    local_manager = SignatureEmbeddingModel(max_students=150)
    
    # ACTUALLY TRAIN THE MODEL (this was missing!)
    logger.info(f"🚀 Starting individual model training for {student_name}")
    training_result = local_manager.train_models(training_data, epochs=settings.MODEL_EPOCHS)
    logger.info(f"✅ Individual model training completed for {student_name}")
    
    # Extract training history for database records
    classification_history = training_result.get('classification_history', {})
    siamese_history = training_result.get('siamese_history', {})

    # Save models - choose between S3 or local storage
    model_uuid = str(uuid.uuid4())
    try:
        # Check if local storage is enabled (override with parameter)
        use_local_storage = not use_s3_upload and os.getenv('USE_LOCAL_STORAGE', 'false').lower() == 'true'
        
        if use_local_storage:
            # Use local storage (INSTANT - no S3 upload)
            from utils.local_model_saving import save_signature_models_locally
            uploaded_files = save_signature_models_locally(
                local_manager, 
                "individual", 
                model_uuid
            )
            logger.info("🚀 Using LOCAL storage (no S3 upload - INSTANT!)")
        else:
            # Use optimized S3 saving with parallel uploads
            uploaded_files = save_signature_models_optimized(
                local_manager, 
                "individual", 
                model_uuid
            )
            logger.info("☁️ Using S3 storage with parallel uploads")
        
        # Extract URLs and keys from uploaded files
        s3_urls = {}
        s3_keys = {}
        for model_type, file_info in uploaded_files.items():
            if model_type in ['embedding', 'classification']:
                s3_urls[model_type] = file_info['url']
                s3_keys[model_type] = file_info['key']
            
        logger.info(f"✅ Individual model {model_uuid} saved with optimized S3 saving")
        
    except Exception as e:
        logger.warning(f"Optimized S3 saving failed, trying direct method: {e}")
        try:
            # Fallback to direct S3 saving
            uploaded_files = save_signature_models_directly(
                local_manager, 
                "individual", 
                model_uuid
            )
            
            # Extract URLs and keys from uploaded files
            s3_urls = {}
            s3_keys = {}
            for model_type, file_info in uploaded_files.items():
                if model_type in ['embedding', 'classification']:
                    s3_urls[model_type] = file_info['url']
                    s3_keys[model_type] = file_info['key']
                
            logger.info(f"✅ Individual model {model_uuid} saved with direct S3 saving")
            
        except Exception as e2:
            logger.error(f"❌ Both optimized and direct S3 saving failed: {e2}")
            # Final fallback to original method
        base_path = os.path.join(settings.LOCAL_MODELS_DIR, f"signature_model_{model_uuid}")
        os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)
        local_manager.save_models(base_path)

        model_files = [
            (f"{base_path}_embedding.keras", "embedding"),
            (f"{base_path}_classification.keras", "classification"),
        ]
        s3_urls = {}
        s3_keys = {}
        for file_path, model_type in model_files:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    model_data = f.read()
                s3_key, s3_url = upload_model_file(model_data, "individual", f"{model_type}_{model_uuid}", "keras")
                s3_urls[model_type] = s3_url
                s3_keys[model_type] = s3_key
                cleanup_local_file(file_path)
        cleanup_local_file(f"{base_path}_mappings.json")

    # Atomic DB write - only create record after successful S3 upload
    model_record = None
    try:
        # Ensure atomic operations
        from utils.s3_supabase_sync import ensure_atomic_operations
        if not await ensure_atomic_operations():
            raise Exception("Database connection not available for atomic operations")
        
        # Verify S3 uploads were successful
        for model_type, file_info in uploaded_files.items():
            if 'key' in file_info:
                from utils.s3_storage import object_exists
                if not object_exists(file_info['key']):
                    raise Exception(f"S3 upload verification failed for {model_type}")
        
        # Upload training logs to S3 for auditability
        logs_url = None
        try:
            import json
            import numpy as np
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            logs_payload = {
                "classification_history": convert_numpy_types(classification_history),
                "siamese_history": convert_numpy_types(siamese_history),
                "student_mappings": {
                    'student_to_id': local_manager.student_to_id,
                    'id_to_student': local_manager.id_to_student
                },
                "created_at": datetime.utcnow().isoformat(),
                "model_uuid": model_uuid,
                "student_id": int(student["id"]),
            }
            logs_bytes = json.dumps(logs_payload).encode("utf-8")
            from utils.s3_storage import upload_model_file as _upload_generic
            _logs_key, logs_url = _upload_generic(logs_bytes, "individual", f"training_logs_{model_uuid}", "json")
        except Exception as e:
            logger.warning(f"Failed to upload training logs: {e}")
        
        # Record in DB (with optional global_model_id linkage)
        # Extract final accuracy from training history
        final_accuracy = None
        if classification_history.get('accuracy'):
            final_accuracy = float(classification_history['accuracy'][-1])
        elif classification_history.get('val_accuracy'):
            final_accuracy = float(classification_history['val_accuracy'][-1])
        
        payload = {
            "student_id": int(student["id"]),
            # Store classification model as primary path for student identification
            "model_path": s3_urls.get("classification", ""),
            "embedding_model_path": s3_urls.get("embedding", ""),
            "s3_key": s3_keys.get("classification", ""),
            "model_uuid": model_uuid,
            "status": "completed",
            "sample_count": len(genuine_arrays) + len(forged_arrays),
            "genuine_count": len(genuine_arrays),
            "forged_count": len(forged_arrays),
            "training_date": datetime.utcnow().isoformat(),
            "accuracy": final_accuracy,  # Store actual accuracy instead of None
            "training_metrics": {
                'model_type': 'ai_signature_verification_individual',
                'architecture': 'signature_embedding_network',
                'epochs_trained': len(classification_history.get('loss', [])),
                'final_accuracy': float(classification_history.get('accuracy', [0])[-1]) if classification_history.get('accuracy') else None,
                'val_accuracy': float(classification_history.get('val_accuracy', [0])[-1]) if classification_history.get('val_accuracy') else None,
                'final_loss': float(classification_history.get('loss', [0])[-1]) if classification_history.get('loss') else None,
                'val_loss': float(classification_history.get('val_loss', [0])[-1]) if classification_history.get('val_loss') else None,
                'embedding_dimension': local_manager.embedding_dim,
            }
        }
        if global_model_id is not None:
            payload["global_model_id"] = int(global_model_id)
        if logs_url:
            # training_logs_path column doesn't exist in DB schema
            # payload["training_logs_path"] = logs_url
            pass
        
        model_record = await db_manager.create_trained_model(payload)
        logger.info(f"✅ Atomic DB write successful for student {student['id']}")
        
    except Exception as e:
        logger.error(f"❌ Atomic DB write failed for student {student['id']}: {e}")
        # If the column global_model_id doesn't exist yet, retry without it
        if "global_model_id" in payload:
            payload.pop("global_model_id", None)
            try:
                model_record = await db_manager.create_trained_model(payload)
                logger.info(f"✅ Atomic DB write successful (retry) for student {student['id']}")
            except Exception as retry_error:
                logger.error(f"❌ Atomic DB write retry failed for student {student['id']}: {retry_error}")
                model_record = None
        else:
            model_record = None

    # Reduce pauses by clearing TF session and triggering GC
    try:
        import gc
        keras.backend.clear_session()
        gc.collect()
    except Exception:
        pass

    return {"record": model_record, "urls": s3_urls}

async def train_signature_model(student, genuine_data, forged_data, job=None):
    """
    Train signature verification model with real AI deep learning
    """
    try:
        if job:
            job_queue.update_job_progress(job.job_id, 5.0, "Initializing AI training system...")
        
        # Create fresh instances for this training session to prevent state contamination
        local_manager = SignatureEmbeddingModel(max_students=150)
        preprocessor = SignaturePreprocessor(target_size=settings.MODEL_IMAGE_SIZE)
        
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

        # Skip forged signature processing - not used for owner identification training
        # (Forgery detection is disabled system-wide - focus on owner identification only)

        if job:
            job_queue.update_job_progress(job.job_id, 50.0, "Preparing training data with augmentation...")

        # Prepare training data with augmentation - use student name for consistent mapping
        student_name = f"{student.get('firstname', '')} {student.get('surname', '')}".strip() or f"Student_{student['id']}"
        training_data = {
            student_name: {
                'genuine': genuine_images,
                'forged': forged_images
            }
        }

        if job:
            job_queue.update_job_progress(job.job_id, 60.0, "Training AI models with deep learning...")

        t0 = time.time()
        
        # Train with classification-only for faster identification
        # Pass job_id to training context for real-time metrics
        if job:
            import threading
            current_thread = threading.current_thread()
            current_thread.job_id = job.job_id
        
        result_models = local_manager.train_models(training_data, epochs=settings.MODEL_EPOCHS)

        if job:
            job_queue.update_job_progress(job.job_id, 85.0, "Saving trained models...")

        model_uuid = str(uuid.uuid4())
        
        # Save models - choose between S3 or local storage
        try:
            # Check if local storage is enabled (override with parameter)
            use_local_storage = not use_s3_upload and os.getenv('USE_LOCAL_STORAGE', 'false').lower() == 'true'
            
            if use_local_storage:
                # Use local storage (INSTANT - no S3 upload)
                from utils.local_model_saving import save_signature_models_locally
                uploaded_files = save_signature_models_locally(
                    local_manager, 
                    "individual", 
                    model_uuid
                )
                logger.info("🚀 Using LOCAL storage (no S3 upload - INSTANT!)")
            else:
                # Use optimized S3 saving with parallel uploads
                uploaded_files = save_signature_models_optimized(
                    local_manager, 
                    "individual", 
                    model_uuid
                )
                logger.info("☁️ Using S3 storage with parallel uploads")
            
            # Extract URLs and KEYS from uploaded files
            s3_urls = {}
            s3_keys = {}
            for model_type, file_info in uploaded_files.items():
                s3_urls[model_type] = file_info.get('url')
                if 'key' in file_info:
                    s3_keys[model_type] = file_info['key']
                
            logger.info(f"✅ Main training model {model_uuid} saved with optimized S3 saving")
            
        except Exception as e:
            logger.warning(f"Optimized S3 saving failed, trying direct method: {e}")
            try:
                # Fallback to direct S3 saving
                uploaded_files = save_signature_models_directly(
                    local_manager, 
                    "individual", 
                    model_uuid
                )
                
                # Extract URLs and KEYS from uploaded files
                s3_urls = {}
                s3_keys = {}
                for model_type, file_info in uploaded_files.items():
                    s3_urls[model_type] = file_info.get('url')
                    if 'key' in file_info:
                        s3_keys[model_type] = file_info['key']
                    
                logger.info(f"✅ Main training model {model_uuid} saved with direct S3 saving")
                
            except Exception as e2:
                logger.error(f"❌ Both optimized and direct S3 saving failed: {e2}")
                # Final fallback to original method
                base_path = os.path.join(settings.LOCAL_MODELS_DIR, f"signature_model_{model_uuid}")
                os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)

            # Save all models
            local_manager.save_models(base_path)

            # Upload models to S3 (only classification and embedding for faster training)
            model_files = [
                (f"{base_path}_embedding.keras", "embedding"),
                (f"{base_path}_classification.keras", "classification")
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
                    s3_keys[model_type] = s3_key
                    
                    # Clean up local file
                    cleanup_local_file(file_path)

            # Clean up mappings file
            cleanup_local_file(f"{base_path}_mappings.json")

        # Upload training logs (metrics) to S3 as JSON for auditability
        try:
            import json
            import numpy as np
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            logs_payload = {
                "classification_history": convert_numpy_types(result_models.get('classification_history', {})),
                "siamese_history": convert_numpy_types(result_models.get('siamese_history', {})),
                "student_mappings": convert_numpy_types(result_models.get('student_mappings', {})),
                "created_at": datetime.utcnow().isoformat(),
                "model_uuid": model_uuid,
                "student_id": int(student["id"]),
            }
            logs_bytes = json.dumps(logs_payload).encode("utf-8")
            from utils.s3_storage import upload_model_file as _upload_generic
            # Reuse model namespace for grouping; store logs alongside models
            _logs_key, logs_url = _upload_generic(logs_bytes, "individual", f"training_logs_{model_uuid}", "json")
        except Exception as e:
            logger.warning(f"Failed to upload training logs: {e}")
            logs_url = None

        # Create model record with comprehensive metrics
        # Prefer classification accuracy; fallback to authenticity, then siamese
        _cls_acc = float(result_models['classification_history'].get('accuracy', [0])[-1]) if 'classification_history' in result_models else None
        _sia_acc = float(result_models['siamese_history'].get('accuracy', [0])[-1]) if 'siamese_history' in result_models else None
        top_level_accuracy = next((a for a in [_cls_acc, _sia_acc] if a is not None), None)
        payload = {
            "student_id": int(student["id"]),
            "model_path": s3_urls.get("classification", ""),
            "embedding_model_path": s3_urls.get("embedding", ""),
            "s3_key": s3_keys.get("classification", ""),
            "status": "completed",
            "sample_count": len(genuine_images) + len(forged_images),
            "genuine_count": len(genuine_images),
            "forged_count": len(forged_images),
            "training_date": datetime.utcnow().isoformat(),
            "accuracy": top_level_accuracy,
            "training_metrics": {
                'model_type': 'ai_signature_verification',
                'architecture': 'signature_embedding_network',
                'student_recognition_accuracy': _cls_acc,
                'siamese_accuracy': _sia_acc,
                'epochs_trained': len(result_models['classification_history'].get('accuracy', [])) if 'classification_history' in result_models else None,
                'embedding_dimension': local_manager.embedding_dim,
                'model_parameters': sum([
                    local_manager.embedding_model.count_params() if local_manager.embedding_model else 0,
                    local_manager.classification_head.count_params() if local_manager.classification_head else 0,
                    local_manager.siamese_model.count_params() if local_manager.siamese_model else 0
                ])
            }
        }
        if logs_url:
            # Optional field; ignore if DB schema lacks it
            # training_logs_path column doesn't exist in DB schema
            # payload["training_logs_path"] = logs_url
            pass
        try:
            model_record = await db_manager.create_trained_model(payload)
        except Exception as e:
            # Retry without optional field if column missing
            # training_logs_path column doesn't exist in DB schema - already removed
            try:
                model_record = await db_manager.create_trained_model(payload)
            except Exception:
                raise e

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
    genuine_files: List[UploadFile] | None = File(None),
    forged_files: List[UploadFile] | None = File(None)
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

        genuine_data: List[bytes] = []
        forged_data: List[bytes] = []
        if genuine_files:  # Only need genuine files for owner identification
            # Use uploaded files
            if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
                raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
            # Forged samples not required since forgery detection is disabled - focus on owner identification only
            genuine_data = [await f.read() for f in genuine_files]
            forged_data = [await f.read() for f in forged_files]
        else:
            # Auto-fetch stored signatures from DB/S3
            rows = await db_manager.list_student_signatures(int(student["id"]))
            if not rows:
                raise HTTPException(status_code=400, detail="No stored signatures available for this student")
            import requests
            for r in rows:
                url = r.get("s3_url")
                label = (r.get("label") or "").lower()
                key = r.get("s3_key") or _derive_s3_key_from_url(url)
                # Prefer S3 download by key
                data: bytes | None = None
                if key:
                    try:
                        data = download_bytes(key)
                    except Exception:
                        data = None
                if data is None:
                    if settings.S3_USE_PRESIGNED_GET and key:
                        try:
                            url = create_presigned_get(key)
                        except Exception:
                            pass
                    try:
                        resp = requests.get(url, timeout=8)
                        resp.raise_for_status()
                        data = resp.content
                    except Exception as e:
                        logger.warning(f"HTTP fetch failed for student {student['id']} key={key}: {e}")
                        continue
                if label == "genuine":
                    genuine_data.append(data)
                else:
                    forged_data.append(data)
            if len(genuine_data) < settings.MIN_GENUINE_SAMPLES:
                raise HTTPException(status_code=400, detail="Insufficient stored signatures to train (need more genuine samples)")

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
    genuine_files: List[UploadFile] | None = File(None),
    forged_files: List[UploadFile] | None = File(None),
    use_gpu: bool = Form(True),
    use_s3_upload: bool = Form(False)
):
    """
    Start AI training on AWS GPU instance for faster training
    
    This endpoint launches an AWS GPU instance, uploads training data,
    trains the AI model on the GPU, and returns the trained model.
    
    Args:
        student_id: Comma-separated student IDs for training
        genuine_files: List of genuine signature image files
        forged_files: List of forged signature image files (not used)
        use_gpu: Whether to use GPU training (default: True)
    
    Returns:
        JSON response with job_id and training status
        
    Features:
        - 10-50x faster than CPU training
        - Real-time progress updates
        - Automatic instance cleanup
        - Same training quality as CPU
    """
    check_database_available()
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

            job = job_queue.create_job(int(student["id"]), "gpu_training")
            # If no files uploaded, auto-fetch from storage
            if not genuine_files or len(genuine_files) == 0:  # Only need genuine files for owner identification
                rows = await db_manager.list_student_signatures(int(student["id"]))
                if not rows:
                    raise HTTPException(status_code=400, detail="No stored signatures available for this student")
                import requests
                genuine_data = []
                forged_data = []
                for r in rows:
                    url = r.get("s3_url"); label = (r.get("label") or "").lower()
                    key = r.get("s3_key") or _derive_s3_key_from_url(url)
                    content: bytes | None = None
                    if key:
                        try:
                            content = download_bytes(key)
                        except Exception:
                            content = None
                    if content is None and url:
                        try:
                            resp = requests.get(url, timeout=8); resp.raise_for_status(); content = resp.content
                        except Exception:
                            continue
                    if not content:
                        continue
                    if label == "genuine": genuine_data.append(content)
                    else: forged_data.append(content)
                if len(genuine_data) < settings.MIN_GENUINE_SAMPLES:
                    raise HTTPException(status_code=400, detail="Insufficient stored signatures to train (need more genuine samples)")
            else:
                if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
                    raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
                # Forged samples not required since forgery detection is disabled - focus on owner identification only
                genuine_data = [await f.read() for f in genuine_files]
                forged_data = [await f.read() for f in forged_files] if forged_files else []
            
            if use_gpu and gpu_training_manager.is_available():
                # Use GPU training
                asyncio.create_task(run_gpu_training(job, student, genuine_data, forged_data, use_s3_upload))
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
            job = job_queue.create_job(0, "global_gpu_training")
            # For global GPU, allow auto-fetch when files are not provided
            if not genuine_files or len(genuine_files) == 0:  # Only need genuine files for owner identification
                genuine_data = []
                forged_data = []
            else:
                if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
                    raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
                # Forged samples not required since forgery detection is disabled - focus on owner identification only
                genuine_data = [await f.read() for f in genuine_files]
                forged_data = [await f.read() for f in forged_files] if forged_files else []
            
            if use_gpu and gpu_training_manager.is_available():
                asyncio.create_task(run_global_gpu_training(job, student_ids, genuine_data, forged_data, use_s3_upload))
                return {
                    "success": True, 
                    "job_id": job.job_id, 
                    "message": "Global GPU training job started", 
                    "stream_url": f"/api/progress/stream/{job.job_id}",
                    "training_type": "global_gpu"
                }
            else:
                asyncio.create_task(run_global_async_training(job, student_ids, genuine_data, forged_data, use_s3_upload))
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
    genuine_files: List[UploadFile] | None = File(None),
    forged_files: List[UploadFile] | None = File(None)
):
    check_database_available()
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

            job = job_queue.create_job(int(student["id"]), "training")
            # Allow auto-fetch when files are not uploaded
            if not genuine_files or len(genuine_files) == 0:  # Only need genuine files for owner identification
                rows = await db_manager.list_student_signatures(int(student["id"]))
                if not rows:
                    raise HTTPException(status_code=400, detail="No stored signatures available for this student")
                import requests
                genuine_data = []
                forged_data = []
                for r in rows:
                    url = r.get("s3_url"); label = (r.get("label") or "").lower()
                    key = r.get("s3_key") or _derive_s3_key_from_url(url)
                    content: bytes | None = None
                    if key:
                        try:
                            content = download_bytes(key)
                        except Exception:
                            content = None
                    if content is None and url:
                        try:
                            resp = requests.get(url, timeout=8); resp.raise_for_status(); content = resp.content
                        except Exception:
                            continue
                    if not content:
                        continue
                    if label == "genuine": genuine_data.append(content)
                    else: forged_data.append(content)
                if len(genuine_data) < settings.MIN_GENUINE_SAMPLES:
                    raise HTTPException(status_code=400, detail="Insufficient stored signatures to train (need more genuine samples)")
            else:
                if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
                    raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
                # Forged samples not required since forgery detection is disabled - focus on owner identification only
                genuine_data = [await f.read() for f in genuine_files]
                forged_data = [await f.read() for f in forged_files] if forged_files else []
            asyncio.create_task(run_async_training(job, student, genuine_data, forged_data))
            return {"success": True, "job_id": job.job_id, "message": "Training job started", "stream_url": f"/api/progress/stream/{job.job_id}"}
        
        else:
            # Multiple students - use global training
            # Create a job for global training
            job = job_queue.create_job(0, "global_training")  # 0 indicates global training
            # Support auto-fetch when files omitted (defer per-student fetch to run_global_async_training)
            if not genuine_files or len(genuine_files) == 0:  # Only need genuine files for owner identification
                genuine_data = []
                forged_data = []
            else:
                if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
                    raise HTTPException(status_code=400, detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required")
                # Forged samples not required since forgery detection is disabled - focus on owner identification only
                genuine_data = [await f.read() for f in genuine_files]
                forged_data = [await f.read() for f in forged_files] if forged_files else []
            asyncio.create_task(run_global_async_training(job, student_ids, genuine_data, forged_data, use_s3_upload))
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
        # Train global classifier with tf.data and validation metrics
        history = gsm.train_global_classifier(data_by_student, epochs=settings.MODEL_EPOCHS)
        
        # Build artifacts for classifier
        model_uuid = str(uuid.uuid4())
        # Build mappings in ID-first format
        ci2sid = {str(ci): sid for ci, sid in getattr(gsm, 'class_index_to_student_id', {}).items()}
        ci2name = {str(ci): str(ci) for ci in getattr(gsm, 'class_index_to_student_id', {}).keys()}
        try:
            # Try to use latest DB student names for nicer mapping (optional)
            for sid, ci in getattr(gsm, 'id_to_class_index', {}).items():
                try:
                    student = await db_manager.get_student(int(sid))
                    if student:
                        name = f"{student.get('firstname','')} {student.get('surname','')}".strip() or student.get('name') or f"Student_{sid}"
                        ci2name[str(ci)] = name
                except Exception:
                    pass
        except Exception:
            pass
        id_first_mappings = {
            "class_index_to_student_id": ci2sid,
            "class_index_to_student_name": ci2name,
            "student_id_to_class_index": {str(v): int(k) for k, v in getattr(gsm, 'class_index_to_student_id', {}).items()}
        }
        # Training results summary
        tr = {
            "final_accuracy": float(history.history.get('accuracy', [0])[-1]) if history.history.get('accuracy') else None,
            "final_val_accuracy": float(history.history.get('val_accuracy', [0])[-1]) if history.history.get('val_accuracy') else None,
            "final_loss": float(history.history.get('loss', [0])[-1]) if history.history.get('loss') else None,
            "final_val_loss": float(history.history.get('val_loss', [0])[-1]) if history.history.get('val_loss') else None,
            "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        }
        spec = build_classifier_spec(num_classes=len(ci2sid), image_size=settings.MODEL_IMAGE_SIZE)
        # Compute centroids for faster verification (optional)
        try:
            centroids_arr = gsm.compute_student_centroids(data_by_student)
            centroids_json = {int(k): v.tolist() for k, v in centroids_arr.items()}
        except Exception as e:
            logger.warning(f"Failed to compute centroids: {e}")
            centroids_json = None
        # Package locally
        artifacts = package_global_classifier_artifacts(
            model_uuid,
            settings.LOCAL_MODELS_DIR,
            getattr(gsm, 'classifier'),
            getattr(gsm, 'embedding_model', None),
            id_first_mappings,
            centroids_json,
            tr,
            spec,
        )
        # Upload artifacts to S3 under models/global/<uuid>
        s3_urls = {}
        s3_keys = {}
        for key_name, path in artifacts.items():
            if not path:
                continue
            with open(path, 'rb') as f:
                ext = os.path.splitext(path)[1].lstrip('.') or 'bin'
                up_key, up_url = upload_file_generic(f.read(), "global", f"{model_uuid}_{key_name}", ext)
                s3_keys[key_name] = up_key
                s3_urls[key_name] = up_url
        
        # Store global model record in dedicated global table
        # Consistently point model_path to classifier SavedModel zip URL
        model_record = await db_manager.create_global_model({
            "model_path": s3_urls.get("savedmodel_zip", ""),
            "s3_key": s3_keys.get("savedmodel_zip", ""),
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
            },
            "mappings_path": s3_urls.get("mappings_path", ""),
            "centroids_path": s3_urls.get("centroids_path", ""),
            "embedding_spec_path": s3_urls.get("classifier_spec_path", "")
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


async def run_gpu_training(job, student, genuine_data, forged_data, use_s3_upload=False):
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

        # Skip forged signature processing - not used for owner identification training
        # (Forgery detection is disabled system-wide - focus on owner identification only)

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
            # Extract accuracy from GPU result if available
            gpu_accuracy = gpu_result.get('accuracy', 0.0)
            if isinstance(gpu_accuracy, (int, float)) and gpu_accuracy > 0:
                accuracy = float(gpu_accuracy)
            else:
                accuracy = None
                
            model_record = await db_manager.create_trained_model({
                "student_id": int(student["id"]),
                "model_path": gpu_result['model_urls'].get('classification', ''),
                "embedding_model_path": gpu_result['model_urls'].get('embedding', ''),
                "status": "completed",
                "sample_count": len(genuine_images) + len(forged_images),
                "genuine_count": len(genuine_images),
                "forged_count": len(forged_images),
                "training_date": datetime.utcnow().isoformat(),
                "accuracy": accuracy,  # Store GPU training accuracy
                "training_metrics": {
                    'model_type': 'ai_signature_verification_gpu',
                    'architecture': 'signature_embedding_network',
                    'training_method': 'aws_gpu_instance',
                    'instance_type': 'g4dn.xlarge',
                    'gpu_acceleration': True,
                    'gpu_accuracy': gpu_accuracy
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

async def run_global_gpu_training(job, student_ids, genuine_data, forged_data, use_s3_upload=False):
    """
    Run global training on AWS GPU instance
    """
    try:
        job_queue.start_job(job.job_id)
        job_queue.update_job_progress(job.job_id, 5.0, "Preparing student list for global GPU training...")

        # Resolve student objects from provided IDs (frontend cards)
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

        # Fetch and validate images from storage for only the selected students
        job_queue.update_job_progress(job.job_id, 12.0, "Fetching stored signatures for selected students...")
        sid_ints = [int(s.get("id")) for s in students]
        logger.info(f"Training with {len(students)} students: {[s.get('firstname', '') + ' ' + s.get('surname', '') for s in students]}")
        logger.info(f"Student IDs: {sid_ints}")
        per_student = await _fetch_and_validate_student_images(sid_ints)
        logger.info(f"Successfully fetched data for {len(per_student)} students: {list(per_student.keys())}")

        # Validate minimum totals across all selected students
        total_genuine = sum(len(v["genuine_images"]) for v in per_student.values())
        total_forged = sum(len(v["forged_images"]) for v in per_student.values())
        if total_genuine < settings.MIN_GENUINE_SAMPLES:
            raise Exception("Insufficient stored signatures across selected students to train global model")

        # Build training data structure for GPU service (expects simple dict of lists of arrays)
        training_data = {}
        for s in students:
            sid = int(s["id"])  # type: ignore[index]
            bucket = per_student.get(sid, {"genuine_images": [], "forged_images": []})
            # Use student name for consistent mapping
            student_name = f"{s.get('firstname', '')} {s.get('surname', '')}".strip() or f"Student_{sid}"
            training_data[student_name] = {
                "genuine_images": bucket["genuine_images"],
                "forged_images": bucket["forged_images"],
            }

        if job:
            job_queue.update_job_progress(job.job_id, 25.0, "Starting global GPU training...")

        # Start GPU training (global)
        gpu_result = await gpu_training_manager.start_gpu_training(
            training_data, job.job_id, 0  # 0 for global training
        )

        if gpu_result['success']:
            if job:
                job_queue.update_job_progress(job.job_id, 90.0, "Global training completed, saving results...")
                logger.info(f"Global GPU training completed for job {job.job_id}")

            # Create global model record
            # Extract accuracy from GPU result if available
            gpu_accuracy = gpu_result.get('accuracy', 0.0)
            if isinstance(gpu_accuracy, (int, float)) and gpu_accuracy > 0:
                accuracy = float(gpu_accuracy)
            else:
                accuracy = None
                
            model_record = await db_manager.create_global_model({
                "model_path": gpu_result['model_urls'].get('classifier_savedmodel_zip', '') or gpu_result['model_urls'].get('classification', ''),
                "s3_key": f"global_models/{job.job_id}",
                "model_uuid": job.job_id,
                "status": "completed",
                "sample_count": int(total_genuine + total_forged),
                "genuine_count": int(total_genuine),
                "forged_count": int(total_forged),
                "student_count": len(students),
                "training_date": datetime.utcnow().isoformat(),
                "accuracy": accuracy,  # Store actual GPU training accuracy
                "training_metrics": {
                    'model_type': 'global_ai_signature_verification_gpu',
                    'architecture': 'signature_embedding_network',
                    'training_method': 'aws_gpu_instance',
                    'instance_type': 'g4dn.xlarge',
                    'gpu_acceleration': True,
                    'gpu_accuracy': gpu_accuracy
                },
                "mappings_path": gpu_result['model_urls'].get('mappings', ''),
                "centroids_path": gpu_result['model_urls'].get('centroids', ''),
                "embedding_spec_path": gpu_result['model_urls'].get('classifier_spec', '')
            })

            result = {
                "success": True,
                "model_id": model_record.get("id") if isinstance(model_record, dict) else None,
                "model_uuid": job.job_id,
                "s3_url": gpu_result['model_urls'].get('classification', ''),
                "student_count": len(students),
                "training_samples": int(total_genuine + total_forged),
                "training_method": "aws_gpu_global",
                "model_urls": gpu_result['model_urls']
            }

            # GLOBAL TRAINING ONLY - No individual training
            if job:
                job_queue.update_job_progress(job.job_id, 100.0, "Global training completed successfully!")
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

async def run_global_async_training(job, student_ids, genuine_data, forged_data, use_s3_upload=False):
    """Run global training for multiple students using uploaded files"""
    try:
        job_queue.start_job(job.job_id)
        job_queue.update_job_progress(job.job_id, 10.0, "Preparing student list...")

        # Resolve students from provided IDs (from UI selected cards)
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

        job_queue.update_job_progress(job.job_id, 20.0, f"Fetching stored signatures for {len(students)} students...")

        sid_ints = [int(s.get("id")) for s in students]
        per_student = await _fetch_and_validate_student_images(sid_ints)

        total_genuine = sum(len(v["genuine_images"]) for v in per_student.values())
        total_forged = sum(len(v["forged_images"]) for v in per_student.values())
        if total_genuine < settings.MIN_GENUINE_SAMPLES:
            raise Exception("Insufficient stored signatures across selected students to train global model")

        training_data = {}
        for s in students:
            sid = int(s["id"])  # type: ignore[index]
            bucket = per_student.get(sid, {"genuine_images": [], "forged_images": []})
            # Use student ID as key (not name) to match GlobalSignatureVerificationModel expectations
            training_data[sid] = {
                'genuine_images': bucket['genuine_images'],
                'forged_images': bucket['forged_images']
            }

        if job:
            job_queue.update_job_progress(job.job_id, 80.0, "Training global model...")

        # Train global model
        gsm = GlobalSignatureVerificationModel()
        history = gsm.train_global_model(training_data)
        
        # Save global model directly to S3 (no local files)
        model_uuid = str(uuid.uuid4())
        try:
            from utils.direct_s3_saving import save_global_model_directly
            s3_key, s3_url = save_global_model_directly(gsm, "global", model_uuid)
            logger.info(f"✅ Global model {model_uuid} saved directly to S3")
        except Exception as e:
            logger.error(f"❌ Failed to save global model directly to S3: {e}")
            # Fallback to local save → upload → cleanup
            base_path = os.path.join(settings.LOCAL_MODELS_DIR, f"global_model_{model_uuid}")
            os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)
            gsm.save_model(f"{base_path}.keras")
            with open(f"{base_path}.keras", 'rb') as f:
                model_data = f.read()
            s3_key, s3_url = upload_model_file(model_data, "global", f"global_{model_uuid}", "keras")
            cleanup_local_file(f"{base_path}.keras")
        
        # Compute and cache centroids for verification speed
        try:
            centroids = gsm.compute_student_centroids(training_data)
            import json
            centroids_bytes = json.dumps({str(k): v.tolist() for k, v in centroids.items()}).encode("utf-8")
            ckey, curl = upload_file_generic(centroids_bytes, "global", f"global_{model_uuid}_centroids", "json")
        except Exception:
            ckey, curl = None, None

        # Store global model record in dedicated global table
        payload = {
            "model_path": s3_url,
            "s3_key": s3_key,
            "model_uuid": model_uuid,
            "status": "completed",
            "sample_count": int(total_genuine + total_forged),
            "genuine_count": int(total_genuine),
            "forged_count": int(total_forged),
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
        }
        if curl:
            payload["centroids_path"] = curl
        model_record = None
        try:
            model_record = await db_manager.create_global_model(payload)
        except Exception as e:
            # Fallback: remove centroids_path if the column doesn't exist
            if "centroids_path" in str(e):
                payload.pop("centroids_path", None)
                model_record = await db_manager.create_global_model(payload)
            else:
                logger.error(f"Failed to create global model record: {e}")
                model_record = None
        
        result = {
            "success": True,
            "model_id": model_record.get("id") if isinstance(model_record, dict) else None,
            "model_uuid": model_uuid,
            "s3_url": s3_url,
            "student_count": len(students),
            "training_samples": int(total_genuine + total_forged)
        }
        
        # Hybrid: also train and store individual models from the already preprocessed arrays (before completing job)
        individual_count = 0
        logger.info(f"Starting individual training for {len(students)} students")
        # GLOBAL TRAINING ONLY - No individual training
        if job:
            job_queue.update_job_progress(job.job_id, 100.0, "Global training completed successfully!")
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
