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
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance
signature_ai_manager = SignatureEmbeddingModel(max_students=150)
preprocessor = SignaturePreprocessor(target_size=settings.MODEL_IMAGE_SIZE)

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
        latest_ai_model = await db_manager.get_latest_ai_model() if hasattr(db_manager, 'get_latest_ai_model') else None
        
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
                raise HTTPException(status_code=500, detail="Failed to load AI models")
        else:
            # Fallback to legacy models
            all_models = await db_manager.get_trained_models()
            if not all_models:
                raise HTTPException(status_code=404, detail="No trained models available")

            # Use latest completed AI model
            eligible = [m for m in all_models if m.get("status") == "completed" and 
                       m.get("training_metrics", {}).get("model_type") == "ai_signature_verification"]
            if not eligible:
                raise HTTPException(status_code=404, detail="No AI models available. Please train a model first.")

            latest_model = max(eligible, key=lambda x: x.get("created_at", ""))
            
            # Load legacy model
            try:
                model_path = latest_model.get("model_path")
                if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                    signature_ai_manager.classification_head = await load_model_from_s3(model_path)
                else:
                    signature_ai_manager.classification_head = await load_model_from_supabase(model_path)
            except Exception as e:
                logger.error(f"Failed to load legacy model: {e}")
                raise HTTPException(status_code=500, detail="Failed to load model")

        # Preprocess test signature with advanced preprocessing
        processed_signature = preprocessor.preprocess_signature(test_image)
        
        # Perform AI verification
        result = signature_ai_manager.verify_signature(processed_signature)
        
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
            "model_type": "ai_signature_verification",
            "ai_architecture": "signature_embedding_network"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI identification failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
        latest_ai_model = await db_manager.get_latest_ai_model() if hasattr(db_manager, 'get_latest_ai_model') else None
        
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
                raise HTTPException(status_code=500, detail="Failed to load AI models")
        else:
            # Fallback to legacy models
            all_models = await db_manager.get_trained_models()
            if not all_models:
                raise HTTPException(status_code=404, detail="No trained models available")

            eligible = [m for m in all_models if m.get("status") == "completed" and 
                       m.get("training_metrics", {}).get("model_type") == "ai_signature_verification"]
            if not eligible:
                raise HTTPException(status_code=404, detail="No AI models available. Please train a model first.")

            latest_model = max(eligible, key=lambda x: x.get("created_at", ""))
            
            try:
                model_path = latest_model.get("model_path")
                if model_path.startswith('https://') and 'amazonaws.com' in model_path:
                    signature_ai_manager.classification_head = await load_model_from_s3(model_path)
                else:
                    signature_ai_manager.classification_head = await load_model_from_supabase(model_path)
            except Exception as e:
                logger.error(f"Failed to load legacy model: {e}")
                raise HTTPException(status_code=500, detail="Failed to load model")

        # Preprocess test signature with advanced preprocessing
        processed_signature = preprocessor.preprocess_signature(test_image)
        
        # Perform AI verification
        result = signature_ai_manager.verify_signature(processed_signature)
        
        # Check if the predicted student matches the target student
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
            "model_type": "ai_signature_verification",
            "ai_architecture": "signature_embedding_network"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI verification failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
