from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
from PIL import Image
import io
import logging

from models.database import db_manager
from models.signature_embedding_model import SignatureEmbeddingModel
from utils.signature_preprocessing import SignaturePreprocessor
from utils.image_processing import validate_image
from utils.storage import load_model_from_supabase, load_model_from_s3
from utils.s3_storage import create_presigned_get, download_bytes
from models.global_signature_model import GlobalSignatureVerificationModel
import requests
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance
signature_ai_manager = SignatureEmbeddingModel(max_students=150)
preprocessor = SignaturePreprocessor(target_size=settings.MODEL_IMAGE_SIZE)

def _get_fallback_response(endpoint_type="identify", student_id=None):
    """Return a fallback response when database is unavailable"""
    base_response = {
        "is_match": False,
        "confidence": 0.0,
        "score": 0.0,
        "global_score": None,
        "student_confidence": 0.0,
        "authenticity_score": 0.0,
        "predicted_student": {
            "id": 0,
            "name": "Database Unavailable"
        },
        "is_unknown": True,
        "model_type": "database_unavailable",
        "ai_architecture": "none",
        "error": "Database connection failed. Please check your Supabase configuration."
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

async def _list_candidate_student_ids(limit: int = 50) -> list[int]:
    """List candidate students that have images (limited)."""
    try:
        items = await db_manager.list_students_with_images()
        ids: list[int] = []
        for it in items:
            sid = it.get("student_id") or it.get("id")
            if isinstance(sid, int):
                ids.append(sid)
            if len(ids) >= limit:
                break
        return ids
    except Exception as e:
        logger.warning(f"Failed to list candidate students: {e}")
        return []

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

        # Try to get latest AI model first, fallback to legacy models
        try:
            latest_ai_model = await db_manager.get_latest_ai_model() if hasattr(db_manager, 'get_latest_ai_model') else None
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            return _get_fallback_response("identify")
        
        if latest_ai_model and latest_ai_model.get("status") == "completed":
            # Use new AI model
            model_paths = {
                'embedding': latest_ai_model.get("embedding_model_path"),
                'classification': latest_ai_model.get("model_path"),
                'authenticity': latest_ai_model.get("authenticity_model_path"),
                'siamese': latest_ai_model.get("siamese_model_path")
            }
            
            # Load AI models
            try:
                for model_type, model_path in model_paths.items():
                    if model_path:
                        if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                            model = await load_model_from_s3(model_path)
                        else:
                            model = await load_model_from_supabase(model_path)
                        
                        # Set the appropriate model
                        if model_type == 'embedding':
                            signature_ai_manager.embedding_model = model
                        elif model_type == 'classification':
                            signature_ai_manager.classification_head = model
                        elif model_type == 'authenticity':
                            signature_ai_manager.authenticity_head = model
                        elif model_type == 'siamese':
                            signature_ai_manager.siamese_model = model
                
                # Load student mappings
                mappings_path = latest_ai_model.get("mappings_path")
                if mappings_path:
                    import json
                    import requests
                    mappings_data = requests.get(mappings_path).json()
                    signature_ai_manager.student_to_id = mappings_data['student_to_id']
                    signature_ai_manager.id_to_student = {int(k): v for k, v in mappings_data['id_to_student'].items()}
                
            except Exception as e:
                logger.error(f"Failed to load AI models: {e}")
                return _get_fallback_response("identify")
        else:
            # Fallback to legacy models
            try:
                all_models = await db_manager.get_trained_models()
            except Exception as e:
                logger.warning(f"Database connection failed: {e}")
                return _get_fallback_response("identify")
            
            if not all_models:
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
                    signature_ai_manager.embedding_model = await (
                        load_model_from_s3(embed_path) if (embed_path.startswith('https://') and 'amazonaws.com' in embed_path) else load_model_from_supabase(embed_path)
                    )
                if auth_path:
                    signature_ai_manager.authenticity_head = await (
                        load_model_from_s3(auth_path) if (auth_path.startswith('https://') and 'amazonaws.com' in auth_path) else load_model_from_supabase(auth_path)
                    )
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
                        signature_ai_manager.classification_head = await (
                            load_model_from_s3(model_path) if (model_path.startswith('https://') and 'amazonaws.com' in model_path) else load_model_from_supabase(model_path)
                        )
            except Exception as e:
                logger.error(f"Failed to load legacy model: {e}")
                return _get_fallback_response("identify")

        # Preprocess test signature with advanced preprocessing
        processed_signature = preprocessor.preprocess_signature(test_image)
        
        # Global-first selection: pick owner using global model, then refine with individual model
        hybrid = {}
        predicted_owner_id = None
        try:
            latest_global = await db_manager.get_latest_global_model() if hasattr(db_manager, 'get_latest_global_model') else None
            if latest_global and latest_global.get("model_path"):
                gsm = GlobalSignatureVerificationModel()
                model_path = latest_global.get("model_path")
                if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                    gsm.load_model(model_path)
                else:
                    model_obj = await load_model_from_supabase(model_path)
                    try:
                        gsm.embedding_model = model_obj
                    except Exception:
                        pass
                # Compute test embedding
                test_emb = gsm.embed_images([processed_signature])[0]
                # Try cached centroids first
                centroids = await _load_cached_centroids(latest_global) or {}
                import numpy as np
                best_sid = None
                best_score = -1.0
                if centroids:
                    for sid, centroid in centroids.items():
                        centroid = np.array(centroid)
                        num = float((test_emb * centroid).sum())
                        den = float((np.linalg.norm(test_emb) * np.linalg.norm(centroid)) + 1e-8)
                        cosine = num / den
                        score01 = (cosine + 1.0) / 2.0
                        if score01 > best_score:
                            best_score = score01
                            best_sid = sid
                else:
                    # Fallback: compute quick centroids online
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
                        score01 = (cosine + 1.0) / 2.0
                        if score01 > best_score:
                            best_score = score01
                            best_sid = sid
                if best_sid is not None:
                    predicted_owner_id = int(best_sid)
                    hybrid["global_score"] = float(best_score)
        except Exception as e:
            logger.warning(f"Global-first selection failed: {e}")

        # Individual model inference to get overall confidence
        try:
            result = signature_ai_manager.verify_signature(processed_signature)
            combined_confidence = result["overall_confidence"]
        except ValueError as e:
            logger.error(f"Model verification failed: {e}")
            return _get_fallback_response("identify")
        except Exception as e:
            logger.error(f"Unexpected error during verification: {e}")
            return _get_fallback_response("identify")
        
        if predicted_owner_id is not None:
            if result.get("predicted_student_id") != predicted_owner_id:
                combined_confidence = float(0.7 * combined_confidence + 0.3 * hybrid.get("global_score", 0.0))
            else:
                combined_confidence = float(0.5 * combined_confidence + 0.5 * hybrid.get("global_score", 0.0))
            result["predicted_student_id"] = predicted_owner_id
        
        return {
            "predicted_student": {
                "id": result["predicted_student_id"],
                "name": result["predicted_student_name"],
            },
            "is_match": result["is_genuine"],
            "confidence": float(combined_confidence),
            "score": float(combined_confidence),
            "global_score": hybrid.get("global_score"),
            "student_confidence": result["student_confidence"],
            "authenticity_score": result["authenticity_score"],
            "is_unknown": result["is_unknown"],
            "model_type": "ai_signature_verification",
            "ai_architecture": "signature_embedding_network"
        }
        
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

        # Use the same AI model loading logic as identify
        try:
            latest_ai_model = await db_manager.get_latest_ai_model() if hasattr(db_manager, 'get_latest_ai_model') else None
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            return _get_fallback_response("verify", student_id)
        
        if latest_ai_model and latest_ai_model.get("status") == "completed":
            # Load AI models (same as identify function)
            model_paths = {
                'embedding': latest_ai_model.get("embedding_model_path"),
                'classification': latest_ai_model.get("model_path"),
                'authenticity': latest_ai_model.get("authenticity_model_path"),
                'siamese': latest_ai_model.get("siamese_model_path")
            }
            
            try:
                for model_type, model_path in model_paths.items():
                    if model_path:
                        if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                            model = await load_model_from_s3(model_path)
                        else:
                            model = await load_model_from_supabase(model_path)
                        
                        if model_type == 'embedding':
                            signature_ai_manager.embedding_model = model
                        elif model_type == 'classification':
                            signature_ai_manager.classification_head = model
                        elif model_type == 'authenticity':
                            signature_ai_manager.authenticity_head = model
                        elif model_type == 'siamese':
                            signature_ai_manager.siamese_model = model
                
                # Load student mappings
                mappings_path = latest_ai_model.get("mappings_path")
                if mappings_path:
                    import json
                    import requests
                    mappings_data = requests.get(mappings_path).json()
                    signature_ai_manager.student_to_id = mappings_data['student_to_id']
                    signature_ai_manager.id_to_student = {int(k): v for k, v in mappings_data['id_to_student'].items()}
                
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
                    signature_ai_manager.classification_head = await load_model_from_s3(model_path)
                else:
                    signature_ai_manager.classification_head = await load_model_from_supabase(model_path)
            except Exception as e:
                logger.error(f"Failed to load legacy model: {e}")
                return _get_fallback_response("verify", student_id)

        # Preprocess test signature with advanced preprocessing
        processed_signature = preprocessor.preprocess_signature(test_image)
        
        # Global-first selection
        hybrid = {}
        predicted_owner_id = None
        try:
            latest_global = await db_manager.get_latest_global_model() if hasattr(db_manager, 'get_latest_global_model') else None
            if latest_global and latest_global.get("model_path"):
                gsm = GlobalSignatureVerificationModel()
                model_path = latest_global.get("model_path")
                if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                    gsm.load_model(model_path)
                else:
                    model_obj = await load_model_from_supabase(model_path)
                    try:
                        gsm.embedding_model = model_obj
                    except Exception:
                        pass
                test_emb = gsm.embed_images([processed_signature])[0]
                centroids = await _load_cached_centroids(latest_global) or {}
                import numpy as np
                best_sid = None
                best_score = -1.0
                if centroids:
                    for sid, centroid in centroids.items():
                        centroid = np.array(centroid)
                        num = float((test_emb * centroid).sum())
                        den = float((np.linalg.norm(test_emb) * np.linalg.norm(centroid)) + 1e-8)
                        cosine = num / den
                        score01 = (cosine + 1.0) / 2.0
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
                        score01 = (cosine + 1.0) / 2.0
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
            result = signature_ai_manager.verify_signature(processed_signature)
            combined_confidence = result["overall_confidence"]
        except ValueError as e:
            logger.error(f"Model verification failed: {e}")
            return _get_fallback_response("verify", student_id)
        except Exception as e:
            logger.error(f"Unexpected error during verification: {e}")
            return _get_fallback_response("verify", student_id)
        
        if predicted_owner_id is not None:
            if result.get("predicted_student_id") != predicted_owner_id:
                combined_confidence = float(0.7 * combined_confidence + 0.3 * hybrid.get("global_score", 0.0))
            else:
                combined_confidence = float(0.5 * combined_confidence + 0.5 * hybrid.get("global_score", 0.0))
            result["predicted_student_id"] = predicted_owner_id
        
        # Check if the predicted student matches the target student
        predicted_student_id = result["predicted_student_id"]
        is_correct_student = (student_id is None) or (predicted_student_id == student_id)
        is_match = is_correct_student and result["is_genuine"]

        return {
            "is_match": is_match,
            "confidence": float(combined_confidence),
            "score": float(combined_confidence),
            "global_score": hybrid.get("global_score"),
            "student_confidence": result["student_confidence"],
            "authenticity_score": result["authenticity_score"],
            "predicted_student": {
                "id": result["predicted_student_id"],
                "name": result["predicted_student_name"],
            },
            "target_student_id": student_id,
            "is_correct_student": is_correct_student,
            "is_genuine": result["is_genuine"],
            "is_unknown": result["is_unknown"],
            "model_type": "ai_signature_verification",
            "ai_architecture": "signature_embedding_network"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI verification failed: {e}")
        return _get_fallback_response("verify", student_id)