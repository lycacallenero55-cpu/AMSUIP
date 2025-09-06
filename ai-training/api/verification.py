from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
from PIL import Image
import io
import logging

from models.database import db_manager
from models.signature_model import SignatureVerificationModel
from utils.image_processing import validate_image
from utils.storage import load_model_from_supabase
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance
signature_ai_manager = SignatureVerificationModel(max_students=150)

@router.post("/identify")
async def identify_signature_owner(
    test_file: UploadFile = File(...)
):
    try:
        if not validate_image(test_file):
            raise HTTPException(status_code=400, detail="Invalid test image")

        test_data = await test_file.read()
        test_image = Image.open(io.BytesIO(test_data))

        all_models = await db_manager.get_trained_models()
        if not all_models:
            raise HTTPException(status_code=404, detail="No trained models available")

        # Use latest completed model tagged for individual recognition
        eligible = [m for m in all_models if m.get("status") == "completed" and m.get("training_metrics", {}).get("model_type") in ("real_ai_individual_recognition", "individual_recognition")]
        if not eligible:
            raise HTTPException(status_code=404, detail="No compatible trained models available")

        latest_model = max(eligible, key=lambda x: x.get("created_at", ""))
        student_model_path = latest_model.get("model_path")
        authenticity_model_path = latest_model.get("embedding_model_path")

        try:
            signature_ai_manager.student_model = await load_model_from_supabase(student_model_path)
            signature_ai_manager.authenticity_model = await load_model_from_supabase(authenticity_model_path)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise HTTPException(status_code=500, detail="Failed to load models")

        result = signature_ai_manager.verify_signature(test_image)
        return {
            "predicted_student": {
                "id": result["predicted_student_id"],
                "name": result["predicted_student_name"],
            },
            "is_match": result["is_genuine"],
            "confidence": result["overall_confidence"],
            "student_confidence": result["student_confidence"],
            "authenticity_score": result["authenticity_score"],
            "is_unknown": result["is_unknown"],
            "model_type": "individual_recognition",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Identification failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/verify")
async def verify_signature(
    test_file: UploadFile = File(...),
    student_id: Optional[int] = None
):
    try:
        if not validate_image(test_file):
            raise HTTPException(status_code=400, detail="Invalid test image")

        test_data = await test_file.read()
        test_image = Image.open(io.BytesIO(test_data))

        all_models = await db_manager.get_trained_models()
        eligible = [m for m in (all_models or []) if m.get("status") == "completed" and m.get("training_metrics", {}).get("model_type") in ("real_ai_individual_recognition", "individual_recognition")]
        if not eligible:
            raise HTTPException(status_code=404, detail="No compatible trained models available")

        latest_model = max(eligible, key=lambda x: x.get("created_at", ""))
        student_model_path = latest_model.get("model_path")
        authenticity_model_path = latest_model.get("embedding_model_path")

        try:
            signature_ai_manager.student_model = await load_model_from_supabase(student_model_path)
            signature_ai_manager.authenticity_model = await load_model_from_supabase(authenticity_model_path)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise HTTPException(status_code=500, detail="Failed to load models")

        result = signature_ai_manager.verify_signature(test_image)
        predicted_student_id = result["predicted_student_id"]
        is_correct_student = (student_id is None) or (predicted_student_id == student_id)
        is_match = is_correct_student and result["is_genuine"]

        return {
            "is_match": is_match,
            "confidence": result["overall_confidence"],
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
            "model_type": "individual_recognition",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
