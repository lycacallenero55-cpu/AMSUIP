from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
from PIL import Image
import io
import logging
from datetime import datetime

from models.database import db_manager
from models.signature_embedding_model import SignatureEmbeddingModel
from utils.signature_preprocessing import SignaturePreprocessor
from utils.image_processing import validate_image
# Removed non-existent imports - using direct S3 download instead
from utils.s3_storage import create_presigned_get, download_bytes
from models.global_signature_model import GlobalSignatureVerificationModel
from utils.s3_supabase_sync import sync_supabase_with_s3, sync_supabase_with_s3_enhanced, get_students_with_missing_images, fix_student_image_counts
import requests
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def verification_health():
    """Health check for verification system"""
    try:
        # Check if we can access the database
        has_db = False
        try:
            await db_manager.get_trained_models()
            has_db = True
        except Exception as e:
            logger.warning(f"Database check failed: {e}")
        
        # Check if we have any trained models
        model_count = 0
        try:
            models = await db_manager.get_trained_models()
            model_count = len(models) if models else 0
        except Exception as e:
            logger.warning(f"Model count check failed: {e}")
        
        return {
            "status": "healthy" if has_db else "degraded",
            "database_available": has_db,
            "trained_models_count": model_count,
            "preprocessor_available": preprocessor is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/debug/models")
async def debug_models():
    """Debug endpoint to check model availability"""
    try:
        # Check database connection
        db_status = "connected"
        try:
            await db_manager.get_trained_models()
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Check for AI models
        ai_model_info = None
        if hasattr(db_manager, 'get_latest_ai_model'):
            try:
                latest_ai = await db_manager.get_latest_ai_model()
                if latest_ai:
                    ai_model_info = {
                        "id": latest_ai.get("id"),
                        "status": latest_ai.get("status"),
                        "embedding_path": latest_ai.get("embedding_model_path"),
                        "classification_path": latest_ai.get("model_path"),
                        "authenticity_path": latest_ai.get("authenticity_model_path"),
                        "mappings_path": latest_ai.get("mappings_path")
                    }
                else:
                    ai_model_info = "No AI models found"
            except Exception as e:
                ai_model_info = f"Error loading AI model: {str(e)}"
        else:
            ai_model_info = "get_latest_ai_model method not available"
        
        # Check legacy models
        legacy_models = []
        try:
            models = await db_manager.get_trained_models()
            if models:
                legacy_models = [{"id": m.get("id"), "status": m.get("status"), "type": m.get("training_metrics", {}).get("model_type")} for m in models[:5]]
        except Exception as e:
            legacy_models = f"Error loading legacy models: {str(e)}"
        
        return {
            "database_status": db_status,
            "ai_model_info": ai_model_info,
            "legacy_models": legacy_models,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Global model instance
signature_ai_manager = SignatureEmbeddingModel(max_students=150)
preprocessor = SignaturePreprocessor(target_size=settings.MODEL_IMAGE_SIZE)

def _get_fallback_response(endpoint_type="identify", student_id=None, error_message="System temporarily unavailable"):
    """Return a fallback response when system is unavailable"""
    base_response = {
        "is_match": False,
        "confidence": 0.0,
        "score": 0.0,
        "global_score": None,
        "student_confidence": 0.0,
        "authenticity_score": 0.0,
        "predicted_student": {
            "id": 0,
            "name": "System Unavailable"
        },
        "is_unknown": True,
        "model_type": "system_unavailable",
        "ai_architecture": "none",
        "error": error_message,
        "success": False
    }
    
    if endpoint_type == "verify":
        base_response.update({
            "target_student_id": student_id,
            "is_correct_student": False,
            "is_genuine": False
        })
    
    return base_response

async def _fetch_genuine_arrays_for_student(student_id: int, max_images: int = 5) -> list:
    """Fetch up to max_images genuine signatures for a student and return preprocessed arrays."""
    try:
        rows = await db_manager.list_student_signatures(student_id)
        arrays = []
        for r in rows:
            if (r.get("label") or "").lower() != "genuine":
                continue
            url = r.get("s3_url")
            key = r.get("s3_key")
            content = None
            if key:
                try:
                    content = download_bytes(key)
                except Exception:
                    content = None
            if content is None and url:
                try:
                    if settings.S3_USE_PRESIGNED_GET and key:
                        url = create_presigned_get(key)
                    resp = requests.get(url, timeout=6)
                    if resp.status_code == 200:
                        content = resp.content
                except Exception:
                    content = None
            if not content:
                continue
            try:
                img = Image.open(io.BytesIO(content)).convert('RGB')
                arr = preprocessor.preprocess_signature(img)
                arrays.append(arr)
            except Exception:
                continue
            if len(arrays) >= max_images:
                break
        return arrays
    except Exception as e:
        logger.warning(f"Failed to fetch genuine arrays for student {student_id}: {e}")
        return []

async def _list_candidate_student_ids(limit: int = 50, allowed_ids: set[int] | None = None) -> list[int]:
    """List candidate students that have images (limited), filtered by allowed_ids if provided."""
    try:
        items = await db_manager.list_students_with_images()
        ids: list[int] = []
        for it in items:
            sid = it.get("student_id") or it.get("id")
            if isinstance(sid, int):
                if allowed_ids is None or int(sid) in allowed_ids:
                    ids.append(sid)
            if len(ids) >= limit:
                break
        return ids
    except Exception as e:
        logger.warning(f"Failed to list candidate students: {e}")
        return []

async def _get_trained_student_ids() -> set[int]:
    try:
        models = await db_manager.get_trained_models()
        if not models:
            return set()
        return {int(m.get("student_id")) for m in models if m.get("status") == "completed" and m.get("student_id") is not None}
    except Exception as e:
        logger.warning(f"Failed to get trained student ids: {e}")
        return set()

async def _load_cached_centroids(latest_global: dict) -> dict | None:
    try:
        curl = latest_global.get("centroids_path")
        if not curl:
            return None
        import requests, json
        resp = requests.get(curl, timeout=8)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {int(k): v for k, v in data.items()}
    except Exception:
        return None

@router.post("/identify")
async def identify_signature_owner(
    test_file: UploadFile = File(...)
):
    """
    AI-powered signature identification with real deep learning
    """
    try:
        if not validate_image(test_file):
            raise HTTPException(status_code=400, detail="Invalid test image")

        test_data = await test_file.read()
        test_image = Image.open(io.BytesIO(test_data))

        # Fast path: try latest global classifier (MobileNetV2) directly
        try:
            latest_global = await db_manager.get_latest_global_model() if hasattr(db_manager, 'get_latest_global_model') else None
            if latest_global and latest_global.get("status") == "completed" and latest_global.get("model_path"):
                model_url = latest_global.get("model_path")
                mappings_url = (
                    latest_global.get("mappings_path")
                    or latest_global.get("mappings_url")
                    or (latest_global.get("training_metrics") or {}).get("mappings_path")
                )
                # Derive fallback mappings URL from model URL if not explicitly stored
                if (not mappings_url) and model_url and 'amazonaws.com' in model_url:
                    try:
                        after = model_url.split('amazonaws.com/', 1)[-1]
                        # Expecting models/<job_id>/classification.keras
                        if after.startswith('models/') and after.endswith('/classification.keras'):
                            base = model_url.rsplit('/', 1)[0]
                            mappings_url = f"{base}/mappings.json"
                            logger.info("Derived mappings_url from model_url fallback")
                    except Exception:
                        pass
                if model_url.startswith('https://') and 'amazonaws.com' in model_url:
                    # Download model via presigned GET
                    s3_key = model_url.split('amazonaws.com/')[-1]
                    from utils.s3_storage import create_presigned_get
                    presigned_url = create_presigned_get(s3_key, expires_seconds=3600)
                    import requests, tempfile, os, json
                    from tensorflow import keras
                    # Download model
                    resp = requests.get(presigned_url, timeout=60)
                    resp.raise_for_status()
                    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                        tmp_model.write(resp.content)
                        model_path_local = tmp_model.name
                    # Download mappings if available
                    id_to_student = {}
                    id_to_name = {}
                    try:
                        if mappings_url:
                            mresp = requests.get(mappings_url, timeout=15)
                            mresp.raise_for_status()
                            mdata = mresp.json()
                            # Preferred explicit mappings from trainer
                            if mdata.get('id_to_student_id'):
                                id_to_student = {int(k): int(v) for k, v in (mdata.get('id_to_student_id') or {}).items()}
                            if mdata.get('id_to_student_name'):
                                id_to_name = {int(k): str(v) for k, v in (mdata.get('id_to_student_name') or {}).items()}
                            # Backward compatibility
                            if not id_to_name and isinstance(mdata.get('students'), list):
                                names = mdata['students']
                                id_to_name = {int(i): str(n) for i, n in enumerate(names)}
                    except Exception:
                        id_to_student = {}
                    try:
                        # Try full-model load first
                        try:
                            classifier = keras.models.load_model(model_path_local, compile=False)
                        except Exception as load_err:
                            logger.warning(f"Keras model load failed, trying weights-only path: {load_err}")
                            # Rebuild simple CNN classifier and load weights
                            num_classes = max(2, len(id_to_student) or 2)
                            import tensorflow as tf
                            inputs = tf.keras.Input(shape=(224, 224, 3))
                            x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
                            x = tf.keras.layers.BatchNormalization()(x)
                            x = tf.keras.layers.MaxPooling2D(2)(x)
                            x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
                            x = tf.keras.layers.BatchNormalization()(x)
                            x = tf.keras.layers.MaxPooling2D(2)(x)
                            x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
                            x = tf.keras.layers.BatchNormalization()(x)
                            x = tf.keras.layers.GlobalAveragePooling2D()(x)
                            x = tf.keras.layers.Dropout(0.5)(x)
                            x = tf.keras.layers.Dense(256, activation='relu')(x)
                            x = tf.keras.layers.Dropout(0.3)(x)
                            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
                            classifier = tf.keras.Model(inputs, outputs)
                            try:
                                # Be tolerant to minor naming/version diffs
                                classifier.load_weights(model_path_local, by_name=True, skip_mismatch=True)
                            except Exception as werr:
                                raise RuntimeError(f"Failed to load weights into reconstructed classifier: {werr}")
                        # Preprocess and predict
                        arr = preprocessor.preprocess_signature(test_image)
                        import numpy as np
                        probs = classifier.predict(np.expand_dims(arr, 0), verbose=0)[0]
                        class_idx = int(np.argmax(probs))
                        confidence = float(np.max(probs))
                        predicted_name = id_to_name.get(class_idx) or f"class_{class_idx}"
                        # Map back to student id if available in mappings
                        predicted_id = int(id_to_student.get(class_idx)) if class_idx in id_to_student else 0
                        # Fallback: resolve ID by name from DB if mapping lacks IDs
                        if (predicted_id == 0) and predicted_name and not predicted_name.startswith("class_"):
                            try:
                                student_response = await db_manager.client.table("students").select("id").eq("name", predicted_name).limit(1).execute()
                                if getattr(student_response, 'data', None):
                                    predicted_id = int(student_response.data[0]['id'])
                                    logger.info(f"Resolved student ID {predicted_id} for name {predicted_name} via DB fallback.")
                            except Exception as db_e:
                                logger.warning(f"Failed to resolve student ID for {predicted_name} from DB: {db_e}")
                        return {
                            "predicted_student": {"id": predicted_id, "name": predicted_name},
                            "is_match": True,
                            "confidence": confidence,
                            "score": confidence,
                            "global_score": confidence,
                            "student_confidence": confidence,
                            "authenticity_score": 0.0,
                            "is_unknown": False,
                            "model_type": "global_classifier",
                            "ai_architecture": "simple_cnn_classifier",
                            "success": True,
                            "decision": "match",
                            "candidates": [
                                {"id": predicted_id, "name": predicted_name, "confidence": confidence}
                            ]
                        }
                    finally:
                        try:
                            os.unlink(model_path_local)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Global classifier fast-path failed: {e}")

        # Try to get latest AI model first, fallback to legacy models
        latest_ai_model = None
        try:
            if hasattr(db_manager, 'get_latest_ai_model'):
                latest_ai_model = await db_manager.get_latest_ai_model()
                logger.info(f"Latest AI model found: {latest_ai_model is not None}")
            else:
                logger.warning("get_latest_ai_model method not available")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return _get_fallback_response("identify", error_message=f"Database error: {str(e)}")
        
        if latest_ai_model and latest_ai_model.get("status") == "completed":
            # Use new AI model
            logger.info(f"Using AI model: {latest_ai_model.get('id')}")
            model_paths = {
                'embedding': latest_ai_model.get("embedding_model_path"),
                'classification': latest_ai_model.get("model_path"),  # May be embedding-only in current mode
                'authenticity': latest_ai_model.get("authenticity_model_path"),
                'siamese': latest_ai_model.get("siamese_model_path")
            }
            logger.info(f"Model paths: {model_paths}")
            
            # Load AI models with proper error handling
            try:
                # Create a fresh model manager for this request
                from models.signature_embedding_model import SignatureEmbeddingModel
                request_model_manager = SignatureEmbeddingModel(max_students=150)
                
                for model_type, model_path in model_paths.items():
                    if not model_path:
                        continue
                        
                    try:
                        # Load model from S3 or Supabase
                        if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                            # Download model from S3
                            import requests
                            import tempfile
                            import os
                            from tensorflow import keras
                            
                            # Extract S3 key from URL and create presigned URL
                            s3_key = model_path.split('amazonaws.com/')[-1]
                            from utils.s3_storage import create_presigned_get
                            presigned_url = create_presigned_get(s3_key, expires_seconds=3600)
                            
                            response = requests.get(presigned_url, timeout=30)
                            response.raise_for_status()
                            
                            # Save to temporary file and load
                            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                                tmp_file.write(response.content)
                                tmp_path = tmp_file.name
                            
                            try:
                                # Try custom deserialization first (for optimized S3 models)
                                try:
                                    from utils.optimized_s3_saving import _deserialize_model_from_bytes
                                    with open(tmp_path, 'rb') as f:
                                        model_data = f.read()
                                    model = _deserialize_model_from_bytes(model_data)
                                    logger.info(f"Loaded model using custom deserialization")
                                except Exception as custom_error:
                                    logger.warning(f"Custom deserialization failed, trying standard Keras: {custom_error}")
                                    model = keras.models.load_model(tmp_path)
                                    logger.info(f"Loaded model using standard Keras")
                                
                                # Set the appropriate model
                                if model_type == 'embedding':
                                    request_model_manager.embedding_model = model
                                elif model_type == 'classification':
                                    # Guard against embedding being loaded as classifier
                                    from numpy import zeros
                                    try:
                                        out = model.predict(zeros((1, request_model_manager.image_size, request_model_manager.image_size, 3)), verbose=0)
                                        if len(out.shape) == 2 and (out.shape[1] <= 1 or out.shape[1] == request_model_manager.embedding_dim):
                                            logger.info("Skipping invalid classification head (1-unit or embedding-sized output)")
                                        else:
                                            request_model_manager.classification_head = model
                                    except Exception:
                                        request_model_manager.classification_head = model
                                elif model_type == 'authenticity':
                                    request_model_manager.authenticity_head = model
                                elif model_type == 'siamese':
                                    request_model_manager.siamese_model = model
                                    
                            finally:
                                # Clean up temp file
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                        else:
                            # Load from Supabase (implement if needed)
                            logger.warning(f"Supabase model loading not implemented for {model_path}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Failed to load {model_type} model from {model_path}: {e}")
                        continue
                
                # Load student mappings
                mappings_path = latest_ai_model.get("mappings_path")
                if mappings_path:
                    try:
                        import json
                        import requests
                        mappings_response = requests.get(mappings_path, timeout=10)
                        mappings_response.raise_for_status()
                        mappings_data = mappings_response.json()
                        request_model_manager.student_to_id = mappings_data['student_to_id']
                        request_model_manager.id_to_student = {int(k): v for k, v in mappings_data['id_to_student'].items()}
                    except Exception as e:
                        logger.error(f"Failed to load student mappings: {e}")
                        # Continue without mappings - will use fallback
                
                # Use the request-specific model manager
                signature_ai_manager = request_model_manager
                
                # CRITICAL FIX: Ensure mappings are loaded
                if not signature_ai_manager.id_to_student:
                    logger.warning("No student mappings loaded, loading from database...")
                    try:
                        # Try to get only the students that were used in training
                        try:
                            # Get students that have signatures (these are the ones used in training)
                            students_with_signatures = await db_manager.list_students_with_images()
                            if students_with_signatures:
                                # Extract unique student IDs from signatures
                                trained_student_ids = set()
                                for student_data in students_with_signatures:
                                    if 'student_id' in student_data:
                                        trained_student_ids.add(student_data['student_id'])
                                
                                logger.info(f"DEBUG: Found {len(trained_student_ids)} students with signatures: {list(trained_student_ids)}")
                                
                                # Get student details for only the trained students
                                students = []
                                for student_id in trained_student_ids:
                                    try:
                                        student_response = db_manager.client.table("students").select("*").eq("id", student_id).execute()
                                        if student_response.data:
                                            student = student_response.data[0]
                                            logger.info(f"DEBUG: Student {student_id} fields: {list(student.keys())}")
                                            
                                            # Try to construct a proper name from available fields
                                            name = None
                                            
                                            # Try to combine firstname and surname (based on the actual field names)
                                            if student.get('firstname') and student.get('surname'):
                                                name = f"{student['firstname']} {student['surname']}"
                                            elif student.get('firstname'):
                                                name = student['firstname']
                                            elif student.get('surname'):
                                                name = student['surname']
                                            # Try other name field combinations
                                            elif student.get('first_name') and student.get('last_name'):
                                                name = f"{student['first_name']} {student['last_name']}"
                                            elif student.get('first_name'):
                                                name = student['first_name']
                                            elif student.get('last_name'):
                                                name = student['last_name']
                                            # Try other name fields
                                            elif student.get('full_name'):
                                                name = student['full_name']
                                            elif student.get('student_name'):
                                                name = student['student_name']
                                            elif student.get('name'):
                                                name = student['name']
                                            # If we have email, try to extract name from it
                                            elif student.get('email'):
                                                email = student['email']
                                                # Extract name from email (before @)
                                                email_name = email.split('@')[0]
                                                # Convert dots and underscores to spaces and capitalize
                                                name = email_name.replace('.', ' ').replace('_', ' ').title()
                                            else:
                                                name = f"Student_{student_id}"
                                            
                                            student['name'] = name
                                            logger.info(f"DEBUG: Student {student_id} name: {name}")
                                            students.append(student)
                                    except Exception as student_error:
                                        logger.warning(f"Failed to get student {student_id}: {student_error}")
                                        continue
                                
                                logger.info(f"DEBUG: Loaded {len(students)} trained students")
                            else:
                                logger.warning("DEBUG: No students with signatures found")
                                students = []
                        except Exception as e:
                            logger.error(f"Failed to get students from students table: {e}")
                            students = []
                        
                        if students:
                            logger.info(f"DEBUG: First student data structure: {students[0] if students else 'None'}")
                            for i, student in enumerate(students):
                                logger.info(f"DEBUG: Student {i}: {student}")
                                student_id = student.get('id')
                                student_name = student.get('name')
                                logger.info(f"DEBUG: Extracted student_id={student_id}, name={student_name}")
                                if student_id and student_name:
                                    # Map model class index (i) to actual student ID
                                    signature_ai_manager.student_to_id[student_name] = i
                                    signature_ai_manager.id_to_student[i] = student_name
                                    # Also map the actual student ID for reference
                                    signature_ai_manager.id_to_student[student_id] = student_name
                                    logger.info(f"DEBUG: Mapped student {student_name} (ID: {student_id}) to class index {i}")
                            logger.info(f"Loaded emergency mappings: {len(signature_ai_manager.id_to_student)} students")
                            # Only log first few mappings to avoid spam
                            sample_mappings = dict(list(signature_ai_manager.id_to_student.items())[:5])
                            logger.info(f"Sample student mappings: {sample_mappings}")
                        else:
                            logger.warning("DEBUG: No students returned from database")
                    except Exception as e:
                        logger.error(f"Failed to load emergency mappings: {e}")
                
            except Exception as e:
                logger.error(f"Failed to load AI models: {e}")
                return _get_fallback_response("identify")
        else:
            # Fallback to legacy models or return error
            logger.warning("No AI model available, checking for legacy models...")
            try:
                all_models = await db_manager.get_trained_models()
                logger.info(f"Found {len(all_models) if all_models else 0} legacy models")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                return _get_fallback_response("identify", error_message=f"Database error: {str(e)}")
            
            if not all_models:
                return _get_fallback_response("identify", error_message="No trained models available. Please train a model first.")

            # Use latest completed AI model (accept individual and gpu variants)
            eligible = [
                m for m in all_models
                if m.get("status") == "completed"
                and (m.get("training_metrics", {}).get("model_type") in (
                    "ai_signature_verification",
                    "ai_signature_verification_individual",
                    "ai_signature_verification_gpu"
                ))
            ]
            if not eligible:
                return {
                    "predicted_student": {
                        "id": 0,
                        "name": "No Model Available"
                    },
                    "is_match": False,
                    "confidence": 0.0,
                    "global_score": None,
                    "student_confidence": 0.0,
                    "authenticity_score": 0.0,
                    "is_unknown": True,
                    "model_type": "no_model_available",
                    "ai_architecture": "none",
                    "error": "No trained models available. Please train a model first."
                }

            latest_model = max(eligible, key=lambda x: x.get("created_at", ""))
            
            # Load legacy/individual model by key first, then URL
            try:
                model_path = latest_model.get("model_path") or ""
                model_key = latest_model.get("s3_key") or None
                # Prefer authenticity/embedding paths if present (per-student training)
                embed_path = latest_model.get("embedding_model_path") or ""
                auth_path = latest_model.get("authenticity_model_path") or ""
                if embed_path:
                    try:
                        if embed_path.startswith('https://') and 'amazonaws.com' in embed_path:
                            # Download from S3
                            import requests
                            import tempfile
                            import os
                            from tensorflow import keras
                            
                            response = requests.get(embed_path, timeout=30)
                            response.raise_for_status()
                            
                            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                                tmp_file.write(response.content)
                                tmp_path = tmp_file.name
                            
                            try:
                                signature_ai_manager.embedding_model = keras.models.load_model(tmp_path)
                            finally:
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                        else:
                            logger.warning(f"Supabase model loading not implemented for {embed_path}")
                    except Exception as e:
                        logger.error(f"Failed to load embedding model: {e}")
                
                if auth_path:
                    try:
                        if auth_path.startswith('https://') and 'amazonaws.com' in auth_path:
                            # Download from S3
                            import requests
                            import tempfile
                            import os
                            from tensorflow import keras
                            
                            response = requests.get(auth_path, timeout=30)
                            response.raise_for_status()
                            
                            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                                tmp_file.write(response.content)
                                tmp_path = tmp_file.name
                            
                            try:
                                signature_ai_manager.authenticity_head = keras.models.load_model(tmp_path)
                            finally:
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                        else:
                            logger.warning(f"Supabase model loading not implemented for {auth_path}")
                    except Exception as e:
                        logger.error(f"Failed to load authenticity model: {e}")
                if model_path and not auth_path and not embed_path:
                    # legacy single path (classification). Try by S3 key first
                    if model_key:
                        from utils.s3_storage import download_model_file
                        from tensorflow import keras
                        import tempfile, os
                        data = download_model_file(model_key)
                        suffix = '.keras' if model_key.endswith('.keras') else '.h5'
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        try:
                            tmp.write(data); tmp.flush(); tmp.close()
                            signature_ai_manager.classification_head = keras.models.load_model(tmp.name)
                        finally:
                            try: os.unlink(tmp.name)
                            except OSError: pass
                    else:
                        # Load model directly from S3
                        if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                            try:
                                import tempfile
                                import os
                                from tensorflow import keras
                                
                                # Extract S3 key from URL and create presigned URL
                                s3_key = model_path.split('amazonaws.com/')[-1]
                                from utils.s3_storage import create_presigned_get
                                presigned_url = create_presigned_get(s3_key, expires_seconds=3600)
                                
                                logger.info(f"Downloading {model_type} model from S3: {s3_key}")
                                response = requests.get(presigned_url, timeout=60)
                                response.raise_for_status()
                                
                                # Verify download size
                                content_length = len(response.content)
                                logger.info(f"Downloaded {model_type} model: {content_length} bytes")
                                
                                if content_length < 1000:  # Models should be much larger
                                    raise ValueError(f"Downloaded file too small: {content_length} bytes")
                                
                                with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                                    tmp_file.write(response.content)
                                    tmp_path = tmp_file.name
                                    
                                logger.info(f"Saved {model_type} model to: {tmp_path}")
                                
                                try:
                                    logger.info(f"Loading {model_type} model from: {tmp_path}")
                                    
                                    # Try custom deserialization first (for optimized S3 models)
                                    try:
                                        from utils.optimized_s3_saving import _deserialize_model_from_bytes
                                        with open(tmp_path, 'rb') as f:
                                            model_data = f.read()
                                        model = _deserialize_model_from_bytes(model_data)
                                        logger.info(f"Loaded {model_type} model using custom deserialization")
                                    except Exception as custom_error:
                                        logger.warning(f"Custom deserialization failed, trying standard Keras: {custom_error}")
                                        model = keras.models.load_model(tmp_path)
                                        logger.info(f"Loaded {model_type} model using standard Keras")
                                    
                                    # Set the appropriate model with safeguards
                                    if model_type == 'embedding':
                                        request_model_manager.embedding_model = model
                                    elif model_type == 'classification':
                                        try:
                                            # Heuristic: route authenticity files to authenticity_head
                                            if 'authenticity' in (model_path or '').lower():
                                                request_model_manager.authenticity_head = model
                                            else:
                                                # If single-unit output, treat as authenticity
                                                output_shape = getattr(model, 'output_shape', None)
                                                if output_shape and isinstance(output_shape, (list, tuple)):
                                                    last_dim = output_shape[-1][-1] if isinstance(output_shape[-1], (list, tuple)) else output_shape[-1]
                                                    if last_dim == 1:
                                                        request_model_manager.authenticity_head = model
                                                    else:
                                                        request_model_manager.classification_head = model
                                                else:
                                                    request_model_manager.classification_head = model
                                        except Exception:
                                            request_model_manager.classification_head = model
                                    elif model_type == 'authenticity':
                                        request_model_manager.authenticity_head = model
                                    elif model_type == 'siamese':
                                        request_model_manager.siamese_model = model
                                    
                                    logger.info(f"Successfully loaded {model_type} model")
                                    
                                except Exception as load_error:
                                    logger.error(f"Failed to load {model_type} model from {tmp_path}: {load_error}")
                                    raise
                                finally:
                                    try:
                                        os.unlink(tmp_path)
                                    except:
                                        pass
                            except Exception as e:
                                logger.error(f"Failed to load classification model: {e}")
                        else:
                            logger.warning(f"Supabase model loading not implemented for {model_path}")
            except Exception as e:
                logger.error(f"Failed to load legacy model: {e}")
                return _get_fallback_response("identify")

        # Preprocess test signature with advanced preprocessing
        processed_signature = preprocessor.preprocess_signature(test_image)
        
        # Global-first selection (restricted to trained students)
        hybrid = {"global_score": 0.0, "global_margin": 0.0, "global_margin_raw": 0.0}
        predicted_owner_id = None
        try:
            trained_ids = await _get_trained_student_ids()
            latest_global = await db_manager.get_latest_global_model() if hasattr(db_manager, 'get_latest_global_model') else None
            if latest_global and latest_global.get("model_path"):
                gsm = GlobalSignatureVerificationModel()
                model_path = latest_global.get("model_path")
                if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                    # Download global model from S3 using presigned URL
                    s3_key = model_path.split('amazonaws.com/')[-1]
                    from utils.s3_storage import create_presigned_get
                    presigned_url = create_presigned_get(s3_key, expires_seconds=3600)
                    
                    import requests
                    import tempfile
                    import os
                    
                    response = requests.get(presigned_url, timeout=30)
                    response.raise_for_status()
                    
                    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name
                    
                    try:
                        gsm.load_model(tmp_path)
                        # Compute test embedding
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                test_emb = gsm.embed_images([processed_signature])[0]
                # Try cached centroids first
                centroids = await _load_cached_centroids(latest_global) or {}
                import numpy as np
                best_sid = None
                best_score = -1.0
                second_best = -1.0
                best_cos = -1.0
                second_cos = -1.0
                if centroids:
                    for sid, centroid in centroids.items():
                        # Restrict to trained students only
                        try:
                            if trained_ids and int(sid) not in trained_ids:
                                continue
                        except Exception:
                            pass
                        centroid = np.array(centroid)
                        num = float((test_emb * centroid).sum())
                        den = float((np.linalg.norm(test_emb) * np.linalg.norm(centroid)) + 1e-8)
                        cosine = num / den
                        # Tuned scaling to avoid near-1.0 inflation
                        score01 = max(0.0, min(1.0, (cosine - 0.5) / 0.5))
                        if score01 > best_score:
                            second_best = best_score
                            second_cos = best_cos
                            best_score = score01
                            best_cos = cosine
                            best_sid = sid
                        elif score01 > second_best:
                            second_best = score01
                            second_cos = cosine
                else:
                    # Fallback: compute quick centroids online (only consider trained IDs if present)
                    candidate_ids = await _list_candidate_student_ids(limit=50, allowed_ids=trained_ids if trained_ids else None)
                    for sid in candidate_ids:
                        arrays = await _fetch_genuine_arrays_for_student(int(sid), max_images=3)
                        if not arrays:
                            continue
                        ref_embs = gsm.embed_images(arrays)
                        centroid = np.mean(ref_embs, axis=0)
                        num = float((test_emb * centroid).sum())
                        den = float((np.linalg.norm(test_emb) * np.linalg.norm(centroid)) + 1e-8)
                        cosine = num / den
                        # Tuned scaling to avoid near-1.0 inflation
                        score01 = max(0.0, min(1.0, (cosine - 0.5) / 0.5))
                        if score01 > best_score:
                            second_best = best_score
                            second_cos = best_cos
                            best_score = score01
                            best_cos = cosine
                            best_sid = sid
                        elif score01 > second_best:
                            second_best = score01
                            second_cos = cosine
                if best_sid is not None:
                    predicted_owner_id = int(best_sid)
                    hybrid["global_score"] = float(best_score)
                    if second_best >= 0:
                        hybrid["global_margin"] = float(best_score - second_best)
                        if second_cos >= -1.0:
                            # Raw cosine margin (0..2 range typical around -1..1)
                            hybrid["global_margin_raw"] = float(best_cos - second_cos)
        except Exception as e:
            logger.warning(f"Global-first selection failed: {e}")

        # Individual model inference to get overall confidence
        try:
            # Check if we have any loaded models
            has_any_model = signature_ai_manager.embedding_model is not None
            
            if not has_any_model:
                logger.error("No models loaded for verification")
                return _get_fallback_response("identify", error_message="No AI models were successfully loaded. Please check if models exist and are accessible.")
            
            result = signature_ai_manager.verify_signature(processed_signature)
            combined_confidence = result["overall_confidence"]
        except ValueError as e:
            logger.error(f"Model verification failed: {e}")
            return _get_fallback_response("identify")
        except Exception as e:
            logger.error(f"Unexpected error during verification: {e}")
            return _get_fallback_response("identify")
        
        # Check if global prediction is valid (has proper margin)
        global_margin = float(hybrid.get("global_margin", 0.0) or 0.0)
        global_margin_raw = float(hybrid.get("global_margin_raw", 0.0) or 0.0)
        accepted_zero_margin_high = False
        
        if predicted_owner_id is not None:
            # Handle zero-margin degeneracy: accept if absolute global score is high enough
            if global_margin <= 1e-6 and global_margin_raw <= 1e-6:
                if hybrid.get("global_score", 0.0) >= 0.95:
                    # Accept the top-1 centroid even with zero margin
                    logger.info(f"Accepting global prediction (high score with zero margin): {predicted_owner_id}")
                    result["predicted_student_id"] = predicted_owner_id
                    result["predicted_student_name"] = signature_ai_manager.id_to_student.get(predicted_owner_id, f"Unknown_{predicted_owner_id}")
                    combined_confidence = float(hybrid.get("global_score", 0.0))
                    accepted_zero_margin_high = True
                else:
                    logger.info(f"Rejecting global prediction due to zero margin: {predicted_owner_id}")
                    predicted_owner_id = None
                    # Keep individual prediction if available
                    individual_prediction = result.get("predicted_student_id", 0)
                    if individual_prediction and individual_prediction > 0:
                        result["predicted_student_id"] = individual_prediction
                        result["predicted_student_name"] = signature_ai_manager.id_to_student.get(individual_prediction, f"Unknown_{individual_prediction}")
                    else:
                        result["predicted_student_id"] = 0
                        result["predicted_student_name"] = "Unknown"
            else:
                individual_prediction = result.get("predicted_student_id")
                logger.info(f"Individual model predicted: {individual_prediction}, Global model predicted: {predicted_owner_id}")
                # Preserve the original individual prediction for agreement checks later
                original_individual_prediction = individual_prediction
                if individual_prediction != predicted_owner_id:
                    logger.info(f"Disagreement between models; not forcing override.")
                    # Keep individual prediction and confidence
                else:
                    logger.info(f"Both models agree on prediction: {predicted_owner_id}")
                    combined_confidence = float(0.5 * combined_confidence + 0.5 * hybrid.get("global_score", 0.0))
                result["predicted_student_id"] = predicted_owner_id
        
        # Apply robust unknown/match logic with configurable confidence threshold
        student_confidence = float(result.get("student_confidence", 0.0))
        global_score = float(hybrid.get("global_score", 0.0) or 0.0)
        global_margin = float(hybrid.get("global_margin", 0.0) or 0.0)
        global_margin_raw = float(hybrid.get("global_margin_raw", 0.0) or 0.0)
        
        # Use configurable confidence threshold
        confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Forgery detection is disabled system-wide - focus on owner identification only
        has_auth = False
        
        # Improved unknown thresholding with configurable confidence threshold
        is_unknown = True
        trained_ids = await _get_trained_student_ids()
        
        # If we already accepted a high-score zero-margin global prediction, finalize early
        if accepted_zero_margin_high and predicted_owner_id is not None and (not trained_ids or int(predicted_owner_id) in trained_ids):
            result["is_unknown"] = False
            is_unknown = False
            ownership_ok = True
            is_match = True
            result["score"] = combined_confidence
            predicted_block = {
                "id": int(predicted_owner_id),
                "name": signature_ai_manager.id_to_student.get(int(predicted_owner_id), f"Unknown_{predicted_owner_id}"),
            }
            response_obj = {
                "predicted_student": predicted_block,
                "is_match": True,
                "confidence": float(combined_confidence),
                "score": float(combined_confidence),
                "global_score": hybrid.get("global_score", 0.0),
                "global_margin": hybrid.get("global_margin", 0.0),
                "student_confidence": result["student_confidence"],
                "authenticity_score": 0.0,  # Always 0 since forgery detection is disabled
                "is_unknown": False,
                "model_type": "ai_signature_verification",
                "ai_architecture": "signature_embedding_network",
                "success": True,
                "decision": "match",
                "candidates": [
                    {
                        "id": int(predicted_owner_id),
                        "name": predicted_block["name"],
                        "confidence": float(combined_confidence)
                    }
                ]
            }
            # Back-compat
            response_obj["predicted_student_id"] = int(predicted_owner_id)
            response_obj["predicted_student_name"] = predicted_block["name"]
            response_obj["is_genuine"] = True  # Always true since we're not doing forgery detection
            logger.info(f"Finalized early due to accepted high-score zero-margin global match: {predicted_owner_id}")
            return response_obj

        # Check if we have a valid prediction first
        if predicted_owner_id is not None and predicted_owner_id > 0:
            # Accept if global is confident AND has good separation
            if global_score >= 0.70 and (global_margin >= 0.05 or global_margin_raw >= 0.02):
                if has_auth:
                    is_unknown = not bool(result.get("is_genuine", False))
                else:
                    is_unknown = False
            # Also accept if individual model is confident (even without global)
            elif student_confidence >= 0.60:
                is_unknown = False
            # Special case: if both models agree on same student, be more lenient
            elif (predicted_owner_id == result.get("predicted_student_id") and 
                  global_score >= 0.60 and student_confidence >= 0.40):
                is_unknown = False
        else:
            # No valid prediction from global model - rely on individual model
            individual_prediction = result.get("predicted_student_id", 0)
            if individual_prediction and individual_prediction > 0:
                # If we have an individual prediction, use it even with low confidence
                # since the global model was rejected
                is_unknown = False
                logger.info(f"Using individual model prediction: {individual_prediction} (global rejected)")
            else:
                is_unknown = True
            
        result["is_unknown"] = is_unknown

        # Use configurable confidence threshold for owner identification
        ownership_ok = (
            student_confidence >= confidence_threshold or
            (global_score >= confidence_threshold + 0.1 and (global_margin >= 0.01 or global_margin_raw >= 0.01)) or
            (student_confidence >= confidence_threshold - 0.1 and global_score >= confidence_threshold)
        )
        
        # Do not auto-accept individual prediction if global was rejected (avoid false positives)
        
        # Forgery detection is disabled system-wide - focus on owner identification only
        auth_ok = True  # Always pass since we're not doing forgery detection

        # Agreement boost: if individual and global agree on the same student, relax unknown
        try:
            # Agreement must be based on the pre-override individual prediction
            agree = (predicted_owner_id is not None and original_individual_prediction == predicted_owner_id)
        except Exception:
            agree = False
        if agree and (global_score >= 0.60 or student_confidence >= 0.30):  # Lowered for minimal signatures
            # If both models agree with at least moderate confidence, treat as known
            result["is_unknown"] = False
            is_unknown = False
            ownership_ok = True  # elevate ownership due to cross-model agreement
            
        # Enhanced outlier detection for untrained students
        # If both models are very uncertain, force unknown regardless of prediction
        if (global_score < 0.55 and student_confidence < 0.40 and 
            (global_margin < 0.02 or global_margin_raw < 0.01)):
            result["is_unknown"] = True
            is_unknown = True
            logger.info(f"Forcing unknown due to low confidence: global={global_score:.3f}, student={student_confidence:.3f}")
            
        # Special case for small student counts (<=2) - be more lenient
        trained_student_count = len(signature_ai_manager.id_to_student) if signature_ai_manager.id_to_student else 0
        if trained_student_count <= 2 and not is_unknown:
            # For very small datasets, relax thresholds significantly
            if (global_score >= 0.50 or student_confidence >= 0.30 or 
                (predicted_owner_id == result.get("predicted_student_id") and global_score >= 0.45)):
                result["is_unknown"] = False
                is_unknown = False
                logger.info(f"Relaxed thresholds for small dataset ({trained_student_count} students)")

        # Mask predictions to trained set only
        if result.get("predicted_student_id") and trained_ids and int(result.get("predicted_student_id")) not in trained_ids:
            is_unknown = True
            ownership_ok = False
        # Final match - must be not unknown AND ownership confirmed
        is_match = (not is_unknown) and ownership_ok and auth_ok

        # k-NN fallback over trained students' genuine embeddings when still unknown or classifier absent
        try:
            need_knn = (not accepted_zero_margin_high) and (is_unknown or (signature_ai_manager.classification_head is None))
            if need_knn:
                import numpy as np
                from collections import defaultdict
                
                # Compute test embedding once using the embedding model
                test_arr = signature_ai_manager._preprocess_signature(processed_signature)
                test_emb = signature_ai_manager.embedding_model.predict(np.expand_dims(test_arr, 0), verbose=0)[0]
                
                # Build reference bank from trained students (genuine only)
                ref_embs = []
                ref_labels = []
                trained_ids_list = sorted(list(trained_ids)) if trained_ids else []
                
                def _derive_s3_key_from_url(url: str) -> str | None:
                    if not url:
                        return None
                    base = url.split('?', 1)[0]
                    if "amazonaws.com" not in base:
                        return None
                    try:
                        parts = base.split(".amazonaws.com/")
                        if len(parts) == 2:
                            return parts[1] or None
                        return base.split("/", 3)[-1] or None
                    except Exception:
                        return None
                
                for sid in trained_ids_list:
                    rows = await db_manager.list_student_signatures(int(sid))
                    taken = 0
                    for r in rows:
                        if (r.get("label") or "").lower() != "genuine":
                            continue
                        url = r.get("s3_url"); key = r.get("s3_key") or _derive_s3_key_from_url(url)
                        content = None
                        if key:
                            try:
                                content = download_bytes(key)
                            except Exception:
                                content = None
                        if content is None and url:
                            try:
                                import requests
                                if settings.S3_USE_PRESIGNED_GET and key:
                                    try:
                                        url = create_presigned_get(key)
                                    except Exception:
                                        pass
                                resp = requests.get(url, timeout=5)
                                if resp.status_code == 200:
                                    content = resp.content
                            except Exception:
                                content = None
                        if not content:
                            continue
                        try:
                            img = Image.open(io.BytesIO(content)).convert('RGB')
                            arr = preprocessor.preprocess_signature(img)
                            emb = signature_ai_manager.embedding_model.predict(np.expand_dims(arr, 0), verbose=0)[0]
                            ref_embs.append(emb)
                            ref_labels.append(int(sid))
                            taken += 1
                            if taken >= 8:
                                break
                        except Exception:
                            continue
                if ref_embs:
                    R = np.vstack(ref_embs)
                    L = np.array(ref_labels)
                    # Cosine similarity
                    num = np.dot(R, test_emb)
                    den = (np.linalg.norm(R, axis=1) * (np.linalg.norm(test_emb) + 1e-8)) + 1e-8
                    sims = num / den
                    # Top-k voting
                    k = int(min(10, len(sims)))
                    idx = np.argsort(-sims)[:k]
                    top_labels = L[idx]
                    top_sims = sims[idx]
                    agg = defaultdict(list)
                    for lab, s in zip(top_labels, top_sims):
                        agg[int(lab)].append(float(s))
                    best_lab = None
                    best_score = -1.0
                    second_score = -1.0
                    for lab, scores in agg.items():
                        sc = float(np.mean(scores))
                        if sc > best_score:
                            second_score = best_score
                            best_score = sc
                            best_lab = lab
                        elif sc > second_score:
                            second_score = sc
                    # Optimized for minimal signatures: more lenient k-NN thresholds
                    if best_lab is not None and best_score >= 0.85 and (best_score - max(second_score, 0.0)) >= 0.01:
                        result["predicted_student_id"] = int(best_lab)
                        result["predicted_student_name"] = signature_ai_manager.id_to_student.get(int(best_lab), f"student_{best_lab}")
                        combined_confidence = max(combined_confidence, float(best_score))
                        is_unknown = False
                        is_match = True
                        ownership_ok = True
                        logger.info(f"kNN fallback selected student {best_lab} with score {best_score:.3f}")
        except Exception as e:
            logger.warning(f"kNN fallback failed: {e}")

        # Remove forced-match shortcut; rely on ownership_ok thresholds only

        # Do not clamp confidence upward
            
        # Ensure score matches confidence for frontend compatibility
        result["score"] = combined_confidence

        # DEBUG: Log comprehensive verification details
        logger.info(f"DEBUG: Verification details - predicted_owner_id={predicted_owner_id}, individual_prediction={result.get('predicted_student_id')}")
        logger.info(f"DEBUG: Confidence scores - student={student_confidence:.3f}, global={global_score:.3f}, auth={result.get('authenticity_score', 0.0):.3f}")
        logger.info(f"DEBUG: Margins - global_margin={global_margin:.3f}, raw_margin={global_margin_raw:.3f}")
        logger.info(f"DEBUG: Decisions - is_unknown={is_unknown}, is_match={is_match}, ownership_ok={ownership_ok}")
        logger.info(f"DEBUG: Final result - predicted_student_id={result.get('predicted_student_id')}, predicted_student_name={result.get('predicted_student_name')}")
        
        # If still unknown and we have trained IDs, return best trained student (owner-of-best-score) as last resort
        if is_unknown:
            try:
                # Choose from: accepted global, kNN best (already applied), or individual prediction
                fallback_id = None
                if predicted_owner_id is not None and (not trained_ids or int(predicted_owner_id) in trained_ids):
                    fallback_id = int(predicted_owner_id)
                elif result.get("predicted_student_id") and (not trained_ids or int(result.get("predicted_student_id")) in trained_ids):
                    fallback_id = int(result.get("predicted_student_id"))
                if fallback_id is not None:
                    result["predicted_student_id"] = fallback_id
                    result["predicted_student_name"] = signature_ai_manager.id_to_student.get(fallback_id, f"student_{fallback_id}")
                    combined_confidence = max(combined_confidence, float(global_score))
            except Exception:
                pass

        # Mask unknowns for UI if thresholds not met - return "no_match" for low confidence
        predicted_block = {
            "id": 0 if is_unknown or not is_match else result["predicted_student_id"],
            "name": "Unknown" if is_unknown or not is_match else result["predicted_student_name"],
        }

        # Build decision and candidates
        decision = "match" if is_match else "no_match"
        top_id = int(result.get("predicted_student_id") or 0)
        top_name = signature_ai_manager.id_to_student.get(top_id, "Unknown") if top_id else "Unknown"
        candidates = []
        if top_id:
            candidates.append({
                "id": top_id,
                "name": top_name,
                "confidence": float(combined_confidence)
            })

        response_obj = {
            "predicted_student": predicted_block,
            "is_match": is_match,
            "confidence": float(combined_confidence),
            "score": float(combined_confidence),
            "global_score": hybrid.get("global_score", 0.0),
            "global_margin": hybrid.get("global_margin", 0.0),
            "student_confidence": result["student_confidence"],
            "authenticity_score": result["authenticity_score"],
            "is_unknown": result["is_unknown"],
            "model_type": "ai_signature_verification",
            "ai_architecture": "signature_embedding_network",
            "success": True,
            "message": "Match found" if is_match else "No match found",
            "decision": decision,
            "candidates": candidates
        }
        # Back-compat fields for UI
        response_obj["predicted_student_id"] = 0 if is_unknown else result["predicted_student_id"]
        response_obj["predicted_student_name"] = "Unknown" if is_unknown else result["predicted_student_name"]
        response_obj["is_genuine"] = is_match
        return response_obj
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI identification failed: {e}")
        return _get_fallback_response("identify")

@router.post("/verify")
async def verify_signature(
    test_file: UploadFile = File(...),
    student_id: Optional[int] = None
):
    """
    AI-powered signature verification with real deep learning
    """
    try:
        if not validate_image(test_file):
            raise HTTPException(status_code=400, detail="Invalid test image")

        test_data = await test_file.read()
        test_image = Image.open(io.BytesIO(test_data))

        # Fast path: try latest global classifier (MobileNetV2) directly
        try:
            latest_global = await db_manager.get_latest_global_model() if hasattr(db_manager, 'get_latest_global_model') else None
            if latest_global and latest_global.get("status") == "completed" and latest_global.get("model_path"):
                model_url = latest_global.get("model_path")
                mappings_url = latest_global.get("mappings_path") or latest_global.get("mappings_url")
                if model_url.startswith('https://') and 'amazonaws.com' in model_url:
                    # Download model via presigned GET
                    s3_key = model_url.split('amazonaws.com/')[-1]
                    from utils.s3_storage import create_presigned_get
                    presigned_url = create_presigned_get(s3_key, expires_seconds=3600)
                    import requests, tempfile, os, json
                    from tensorflow import keras
                    # Download model
                    resp = requests.get(presigned_url, timeout=60)
                    resp.raise_for_status()
                    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
                        tmp_model.write(resp.content)
                        model_path_local = tmp_model.name
                    # Download mappings if available
                    id_to_student = {}
                    try:
                        if mappings_url:
                            mresp = requests.get(mappings_url, timeout=15)
                            mresp.raise_for_status()
                            mdata = mresp.json()
                            id_to_student = {int(k): v for k, v in (mdata.get('id_to_student') or {}).items()}
                    except Exception:
                        id_to_student = {}
                    try:
                        # Try full-model load first
                        try:
                            classifier = keras.models.load_model(model_path_local, compile=False)
                        except Exception as load_err:
                            logger.warning(f"Keras model load failed (verify), trying weights-only path: {load_err}")
                            # Rebuild MobileNetV2 classifier and load weights
                            num_classes = 2
                            try:
                                # If mappings present, set num_classes accordingly
                                num_classes = max(2, len(id_to_student) or 2)
                            except Exception:
                                pass
                            import tensorflow as tf
                            base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(settings.MODEL_IMAGE_SIZE, settings.MODEL_IMAGE_SIZE, 3))
                            x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
                            out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
                            classifier = tf.keras.Model(inputs=base.input, outputs=out)
                            try:
                                classifier.load_weights(model_path_local)
                            except Exception as werr:
                                raise RuntimeError(f"Failed to load weights into reconstructed classifier: {werr}")
                        # Preprocess and predict
                        arr = preprocessor.preprocess_signature(test_image)
                        import numpy as np
                        probs = classifier.predict(np.expand_dims(arr, 0), verbose=0)[0]
                        class_idx = int(np.argmax(probs))
                        confidence = float(np.max(probs))
                        predicted_name = id_to_student.get(class_idx, f"class_{class_idx}")
                        predicted_id = class_idx if class_idx in id_to_student else 0
                        is_correct_student = (student_id is None) or (predicted_id == student_id)
                        return {
                            "is_match": is_correct_student,
                            "confidence": confidence,
                            "score": confidence,
                            "global_score": confidence,
                            "student_confidence": confidence,
                            "authenticity_score": 0.0,
                            "predicted_student": {"id": predicted_id, "name": predicted_name},
                            "target_student_id": student_id,
                            "is_correct_student": is_correct_student,
                            "is_genuine": True,
                            "is_unknown": False,
                            "model_type": "global_classifier",
                            "ai_architecture": "simple_cnn_classifier",
                            "success": True,
                            "message": "Match found" if is_correct_student else "No match found",
                            "decision": "match" if is_correct_student else "no_match",
                            "candidates": [
                                {"id": predicted_id, "name": predicted_name, "confidence": confidence}
                            ]
                        }
                    finally:
                        try:
                            os.unlink(model_path_local)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Global classifier fast-path (verify) failed: {e}")

        # Use the same AI model loading logic as identify
        latest_ai_model = None
        try:
            if hasattr(db_manager, 'get_latest_ai_model'):
                latest_ai_model = await db_manager.get_latest_ai_model()
                logger.info(f"Latest AI model found for verify: {latest_ai_model is not None}")
            else:
                logger.warning("get_latest_ai_model method not available")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return _get_fallback_response("verify", student_id, error_message=f"Database error: {str(e)}")
        
        if latest_ai_model and latest_ai_model.get("status") == "completed":
            # Load AI models (same as identify function)
            model_paths = {
                'embedding': latest_ai_model.get("embedding_model_path"),
                'classification': latest_ai_model.get("model_path"),  # Classification model is stored in model_path
                'authenticity': latest_ai_model.get("authenticity_model_path"),
                'siamese': latest_ai_model.get("siamese_model_path")
            }
            
            try:
                # Create a fresh model manager for this request
                from models.signature_embedding_model import SignatureEmbeddingModel
                request_model_manager = SignatureEmbeddingModel(max_students=150)
                
                for model_type, model_path in model_paths.items():
                    if not model_path:
                        continue
                        
                    try:
                        # Load model from S3 or Supabase
                        if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                            # Download model from S3
                            import requests
                            import tempfile
                            import os
                            from tensorflow import keras
                            
                            # Extract S3 key from URL and create presigned URL
                            s3_key = model_path.split('amazonaws.com/')[-1]
                            from utils.s3_storage import create_presigned_get
                            presigned_url = create_presigned_get(s3_key, expires_seconds=3600)
                            
                            response = requests.get(presigned_url, timeout=30)
                            response.raise_for_status()
                            
                            # Save to temporary file and load
                            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                                tmp_file.write(response.content)
                                tmp_path = tmp_file.name
                            
                            try:
                                # Try custom deserialization first (for optimized S3 models)
                                try:
                                    from utils.optimized_s3_saving import _deserialize_model_from_bytes
                                    with open(tmp_path, 'rb') as f:
                                        model_data = f.read()
                                    model = _deserialize_model_from_bytes(model_data)
                                    logger.info(f"Loaded model using custom deserialization")
                                except Exception as custom_error:
                                    logger.warning(f"Custom deserialization failed, trying standard Keras: {custom_error}")
                                    model = keras.models.load_model(tmp_path)
                                    logger.info(f"Loaded model using standard Keras")
                                
                                # Set the appropriate model
                                if model_type == 'embedding':
                                    request_model_manager.embedding_model = model
                                elif model_type == 'classification':
                                    request_model_manager.classification_head = model
                                elif model_type == 'authenticity':
                                    request_model_manager.authenticity_head = model
                                elif model_type == 'siamese':
                                    request_model_manager.siamese_model = model
                                    
                            finally:
                                # Clean up temp file
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                        else:
                            # Load from Supabase (implement if needed)
                            logger.warning(f"Supabase model loading not implemented for {model_path}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Failed to load {model_type} model from {model_path}: {e}")
                        continue
                
                # Load student mappings
                mappings_path = latest_ai_model.get("mappings_path")
                if mappings_path:
                    try:
                        import json
                        import requests
                        mappings_response = requests.get(mappings_path, timeout=10)
                        mappings_response.raise_for_status()
                        mappings_data = mappings_response.json()
                        request_model_manager.student_to_id = mappings_data['student_to_id']
                        request_model_manager.id_to_student = {int(k): v for k, v in mappings_data['id_to_student'].items()}
                    except Exception as e:
                        logger.error(f"Failed to load student mappings: {e}")
                        # Continue without mappings - will use fallback
                
                # Use the request-specific model manager
                signature_ai_manager = request_model_manager
                
                # CRITICAL FIX: Ensure mappings are loaded
                if not signature_ai_manager.id_to_student:
                    logger.warning("No student mappings loaded, loading from database...")
                    try:
                        # Try to get only the students that were used in training
                        try:
                            # Get students that have signatures (these are the ones used in training)
                            students_with_signatures = await db_manager.list_students_with_images()
                            if students_with_signatures:
                                # Extract unique student IDs from signatures
                                trained_student_ids = set()
                                for student_data in students_with_signatures:
                                    if 'student_id' in student_data:
                                        trained_student_ids.add(student_data['student_id'])
                                
                                logger.info(f"DEBUG: Found {len(trained_student_ids)} students with signatures: {list(trained_student_ids)}")
                                
                                # Get student details for only the trained students
                                students = []
                                for student_id in trained_student_ids:
                                    try:
                                        student_response = db_manager.client.table("students").select("*").eq("id", student_id).execute()
                                        if student_response.data:
                                            student = student_response.data[0]
                                            logger.info(f"DEBUG: Student {student_id} fields: {list(student.keys())}")
                                            
                                            # Try to construct a proper name from available fields
                                            name = None
                                            
                                            # Try to combine firstname and surname (based on the actual field names)
                                            if student.get('firstname') and student.get('surname'):
                                                name = f"{student['firstname']} {student['surname']}"
                                            elif student.get('firstname'):
                                                name = student['firstname']
                                            elif student.get('surname'):
                                                name = student['surname']
                                            # Try other name field combinations
                                            elif student.get('first_name') and student.get('last_name'):
                                                name = f"{student['first_name']} {student['last_name']}"
                                            elif student.get('first_name'):
                                                name = student['first_name']
                                            elif student.get('last_name'):
                                                name = student['last_name']
                                            # Try other name fields
                                            elif student.get('full_name'):
                                                name = student['full_name']
                                            elif student.get('student_name'):
                                                name = student['student_name']
                                            elif student.get('name'):
                                                name = student['name']
                                            # If we have email, try to extract name from it
                                            elif student.get('email'):
                                                email = student['email']
                                                # Extract name from email (before @)
                                                email_name = email.split('@')[0]
                                                # Convert dots and underscores to spaces and capitalize
                                                name = email_name.replace('.', ' ').replace('_', ' ').title()
                                            else:
                                                name = f"Student_{student_id}"
                                            
                                            student['name'] = name
                                            logger.info(f"DEBUG: Student {student_id} name: {name}")
                                            students.append(student)
                                    except Exception as student_error:
                                        logger.warning(f"Failed to get student {student_id}: {student_error}")
                                        continue
                                
                                logger.info(f"DEBUG: Loaded {len(students)} trained students")
                            else:
                                logger.warning("DEBUG: No students with signatures found")
                                students = []
                        except Exception as e:
                            logger.error(f"Failed to get students from students table: {e}")
                            students = []
                        
                        if students:
                            logger.info(f"DEBUG: First student data structure: {students[0] if students else 'None'}")
                            for i, student in enumerate(students):
                                logger.info(f"DEBUG: Student {i}: {student}")
                                student_id = student.get('id')
                                student_name = student.get('name')
                                logger.info(f"DEBUG: Extracted student_id={student_id}, name={student_name}")
                                if student_id and student_name:
                                    # Map model class index (i) to actual student ID
                                    signature_ai_manager.student_to_id[student_name] = i
                                    signature_ai_manager.id_to_student[i] = student_name
                                    # Also map the actual student ID for reference
                                    signature_ai_manager.id_to_student[student_id] = student_name
                                    logger.info(f"DEBUG: Mapped student {student_name} (ID: {student_id}) to class index {i}")
                            logger.info(f"Loaded emergency mappings: {len(signature_ai_manager.id_to_student)} students")
                            # Only log first few mappings to avoid spam
                            sample_mappings = dict(list(signature_ai_manager.id_to_student.items())[:5])
                            logger.info(f"Sample student mappings: {sample_mappings}")
                        else:
                            logger.warning("DEBUG: No students returned from database")
                    except Exception as e:
                        logger.error(f"Failed to load emergency mappings: {e}")
                
            except Exception as e:
                logger.error(f"Failed to load AI models: {e}")
                return _get_fallback_response("verify", student_id)
        else:
            # Fallback to legacy models
            try:
                all_models = await db_manager.get_trained_models()
            except Exception as e:
                logger.warning(f"Database connection failed: {e}")
                return _get_fallback_response("verify", student_id)

            if not all_models:
                return _get_fallback_response("verify", student_id)

            eligible = [m for m in all_models if m.get("status") == "completed" and 
                       m.get("training_metrics", {}).get("model_type") == "ai_signature_verification"]
            if not eligible:
                return _get_fallback_response("verify", student_id)

            latest_model = max(eligible, key=lambda x: x.get("created_at", ""))
            
            try:
                model_path = latest_model.get("model_path")
                if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                    # Load model directly from S3
                    if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                        try:
                            import tempfile
                            import os
                            from tensorflow import keras
                            
                            # Extract S3 key from URL and create presigned URL
                            s3_key = model_path.split('amazonaws.com/')[-1]
                            from utils.s3_storage import create_presigned_get
                            presigned_url = create_presigned_get(s3_key, expires_seconds=3600)
                            
                            response = requests.get(presigned_url, timeout=30)
                            response.raise_for_status()
                            
                            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                                tmp_file.write(response.content)
                                tmp_path = tmp_file.name
                            
                            try:
                                signature_ai_manager.classification_head = keras.models.load_model(tmp_path)
                            finally:
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                        except Exception as e:
                            logger.error(f"Failed to load classification model: {e}")
                    else:
                        logger.warning(f"Supabase model loading not implemented for {model_path}")
                else:
                    logger.warning(f"Supabase model loading not implemented for {model_path}")
            except Exception as e:
                logger.error(f"Failed to load legacy model: {e}")
                return _get_fallback_response("verify", student_id)

        # Preprocess test signature with advanced preprocessing
        processed_signature = preprocessor.preprocess_signature(test_image)
        
        # Global-first selection
        hybrid = {"global_score": 0.0, "global_margin": 0.0, "global_margin_raw": 0.0}
        predicted_owner_id = None
        try:
            latest_global = await db_manager.get_latest_global_model() if hasattr(db_manager, 'get_latest_global_model') else None
            if latest_global and latest_global.get("model_path"):
                gsm = GlobalSignatureVerificationModel()
                model_path = latest_global.get("model_path")
                if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                    # Download global model from S3 using presigned URL
                    s3_key = model_path.split('amazonaws.com/')[-1]
                    from utils.s3_storage import create_presigned_get
                    presigned_url = create_presigned_get(s3_key, expires_seconds=3600)
                    
                    import requests
                    import tempfile
                    import os
                    
                    response = requests.get(presigned_url, timeout=30)
                    response.raise_for_status()
                    
                    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name
                    
                    try:
                        gsm.load_model(tmp_path)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                else:
                    logger.warning(f"Supabase model loading not implemented for global model: {model_path}")
                    # Skip global model if not S3
                    return _get_fallback_response("identify", error_message="Global model not available")
                
                test_emb = gsm.embed_images([processed_signature])[0]
                centroids = await _load_cached_centroids(latest_global) or {}
                import numpy as np
                best_sid = None
                best_score = -1.0
                if centroids:
                    for sid, centroid in centroids.items():
                        # Restrict to trained students only
                        try:
                            if trained_ids and int(sid) not in trained_ids:
                                continue
                        except Exception:
                            pass
                        centroid = np.array(centroid)
                        num = float((test_emb * centroid).sum())
                        den = float((np.linalg.norm(test_emb) * np.linalg.norm(centroid)) + 1e-8)
                        cosine = num / den
                        # Tuned scaling to avoid near-1.0 inflation
                        score01 = max(0.0, min(1.0, (cosine - 0.5) / 0.5))
                        if score01 > best_score:
                            best_score = score01
                            best_sid = sid
                else:
                    candidate_ids = await _list_candidate_student_ids(limit=50)
                    for sid in candidate_ids:
                        arrays = await _fetch_genuine_arrays_for_student(int(sid), max_images=3)
                        if not arrays:
                            continue
                        ref_embs = gsm.embed_images(arrays)
                        centroid = np.mean(ref_embs, axis=0)
                        num = float((test_emb * centroid).sum())
                        den = float((np.linalg.norm(test_emb) * np.linalg.norm(centroid)) + 1e-8)
                        cosine = num / den
                        # Tuned scaling to avoid near-1.0 inflation
                        score01 = max(0.0, min(1.0, (cosine - 0.5) / 0.5))
                        if score01 > best_score:
                            best_score = score01
                            best_sid = sid
                if best_sid is not None:
                    predicted_owner_id = int(best_sid)
                    hybrid["global_score"] = float(best_score)
        except Exception as e:
            logger.warning(f"Global-first selection failed: {e}")

        # Individual model inference
        try:
            # Check if we have any loaded models
            has_any_model = (
                signature_ai_manager.embedding_model is not None or
                signature_ai_manager.classification_head is not None or
                signature_ai_manager.authenticity_head is not None
            )
            
            if not has_any_model:
                logger.warning("No models loaded for verification")
                return _get_fallback_response("verify", student_id)
            
            result = signature_ai_manager.verify_signature(processed_signature)
            combined_confidence = result["overall_confidence"]
        except ValueError as e:
            logger.error(f"Model verification failed: {e}")
            return _get_fallback_response("verify", student_id)
        except Exception as e:
            logger.error(f"Unexpected error during verification: {e}")
            return _get_fallback_response("verify", student_id)
        
        # Check if global prediction is valid (has proper margin)
        global_margin = float(hybrid.get("global_margin", 0.0) or 0.0)
        global_margin_raw = float(hybrid.get("global_margin_raw", 0.0) or 0.0)
        
        if predicted_owner_id is not None:
            # Reject global predictions with zero margin (degenerate case)
            if global_margin <= 1e-6 and global_margin_raw <= 1e-6:
                logger.info(f"Rejecting global prediction due to zero margin: {predicted_owner_id}")
                predicted_owner_id = None
                # Keep individual prediction if available
                individual_prediction = result.get("predicted_student_id", 0)
                if individual_prediction and individual_prediction > 0:
                    result["predicted_student_id"] = individual_prediction
                    result["predicted_student_name"] = signature_ai_manager.id_to_student.get(individual_prediction, f"Unknown_{individual_prediction}")
                else:
                    result["predicted_student_id"] = 0
                    result["predicted_student_name"] = "Unknown"
            else:
                individual_prediction = result.get("predicted_student_id")
                logger.info(f"Individual model predicted: {individual_prediction}, Global model predicted: {predicted_owner_id}")
                if individual_prediction != predicted_owner_id:
                    logger.info(f"Using global model prediction: {predicted_owner_id} (overriding individual: {individual_prediction})")
                    combined_confidence = float(0.7 * combined_confidence + 0.3 * hybrid.get("global_score", 0.0))
                    # Update the predicted student name to match the new ID
                    result["predicted_student_name"] = signature_ai_manager.id_to_student.get(predicted_owner_id, f"Unknown_{predicted_owner_id}")
                    logger.info(f"Updated predicted student name to: {result['predicted_student_name']}")
                else:
                    logger.info(f"Both models agree on prediction: {predicted_owner_id}")
                    combined_confidence = float(0.5 * combined_confidence + 0.5 * hybrid.get("global_score", 0.0))
                result["predicted_student_id"] = predicted_owner_id
        
        # Apply robust unknown/match logic with configurable confidence threshold
        student_confidence = float(result.get("student_confidence", 0.0))
        global_score = float(hybrid.get("global_score", 0.0) or 0.0)
        global_margin = float(hybrid.get("global_margin", 0.0) or 0.0)
        global_margin_raw = float(hybrid.get("global_margin_raw", 0.0) or 0.0)
        
        # Use configurable confidence threshold
        confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Forgery detection is disabled system-wide - focus on owner identification only
        has_auth = False
        
        # Improved unknown thresholding with configurable confidence threshold
        is_unknown = True
        
        # Check if we have a valid prediction first
        predicted_student_id = result.get("predicted_student_id", 0)
        if predicted_student_id and predicted_student_id > 0:
            # Accept if global is confident AND has good separation
            if global_score >= 0.70 and (global_margin >= 0.05 or global_margin_raw >= 0.02):
                is_unknown = False
            # Also accept if individual model is confident (even without global)
            elif student_confidence >= confidence_threshold:
                is_unknown = False
            # Special case: if both models agree on same student, be more lenient
            elif (predicted_student_id == result.get("predicted_student_id") and 
                  global_score >= 0.60 and student_confidence >= 0.40):
                is_unknown = False
        else:
            # No valid prediction from global model - rely on individual model
            individual_prediction = result.get("predicted_student_id", 0)
            if individual_prediction and individual_prediction > 0:
                # If we have an individual prediction, use it even with low confidence
                # since the global model was rejected
                is_unknown = False
                logger.info(f"Using individual model prediction: {individual_prediction} (global rejected)")
            else:
                is_unknown = True
        
        result["is_unknown"] = is_unknown

        # Check if the predicted student matches the target student
        is_correct_student = (student_id is None) or (predicted_student_id == student_id)
        ownership_ok = (
            student_confidence >= confidence_threshold or
            (global_score >= confidence_threshold + 0.1 and (global_margin >= 0.01 or global_margin_raw >= 0.01)) or
            (student_confidence >= confidence_threshold - 0.1 and global_score >= confidence_threshold)
        )
        
        # Forgery detection is disabled system-wide - focus on owner identification only
        auth_ok = True  # Always pass since we're not doing forgery detection
        is_match = is_correct_student and (not is_unknown) and ownership_ok

        # Mask unknowns in response - return "no_match" for low confidence
        predicted_block = {
            "id": 0 if is_unknown or not is_match else result["predicted_student_id"],
            "name": "Unknown" if is_unknown or not is_match else result["predicted_student_name"],
        }

        decision = "match" if is_match else "no_match"
        top_id = int(result.get("predicted_student_id") or 0)
        top_name = signature_ai_manager.id_to_student.get(top_id, "Unknown") if top_id else "Unknown"
        candidates = []
        if top_id:
            candidates.append({
                "id": top_id,
                "name": top_name,
                "confidence": float(combined_confidence)
            })

        return {
            "is_match": is_match,
            "confidence": float(combined_confidence),
            "score": float(combined_confidence),
            "global_score": hybrid.get("global_score", 0.0),
            "student_confidence": result["student_confidence"],
            "authenticity_score": result["authenticity_score"],
            "predicted_student": predicted_block,
            "target_student_id": student_id,
            "is_correct_student": is_correct_student,
            "is_genuine": result["is_genuine"],
            "is_unknown": result["is_unknown"],
            "model_type": "ai_signature_verification",
            "ai_architecture": "signature_embedding_network",
            "success": True,
            "message": "Match found" if is_match else "No match found",
            "decision": decision,
            "candidates": candidates
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI verification failed: {e}")
        return _get_fallback_response("verify", student_id)

@router.post("/sync-s3-supabase")
async def sync_s3_supabase(dry_run: bool = True):
    """
    Synchronize Supabase records with S3 objects using enhanced sync
    """
    try:
        result = await sync_supabase_with_s3_enhanced(dry_run=dry_run)
        return {
            "success": True,
            "dry_run": dry_run,
            "sync_stats": result,
            "message": "Enhanced sync completed successfully" if not dry_run else "Enhanced dry run completed - no changes made"
        }
    except Exception as e:
        logger.error(f"Enhanced S3-Supabase sync failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Enhanced sync failed"
        }

@router.get("/missing-images")
async def get_missing_images():
    """
    Get list of students with missing S3 images
    """
    try:
        missing_images = await get_students_with_missing_images()
        return {
            "success": True,
            "missing_images": missing_images,
            "count": len(missing_images)
        }
    except Exception as e:
        logger.error(f"Failed to get missing images: {e}")
        return {
            "success": False,
            "error": str(e),
            "missing_images": []
        }

@router.post("/fix-student-images/{student_id}")
async def fix_student_images(student_id: int):
    """
    Fix image counts for a specific student
    """
    try:
        result = await fix_student_image_counts(student_id)
        return {
            "success": True,
            "student_id": student_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to fix images for student {student_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "student_id": student_id
        }
