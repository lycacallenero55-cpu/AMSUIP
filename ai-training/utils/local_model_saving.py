"""
Local Model Saving Utilities
Save models locally instead of uploading to S3 for faster development
"""

import os
import json
import logging
import time
from typing import Dict, Any, Tuple
from datetime import datetime
from tensorflow import keras
import uuid

logger = logging.getLogger(__name__)

class LocalModelSaver:
    """Save models locally instead of S3"""
    
    def __init__(self, model_type: str, model_uuid: str = None):
        self.model_type = model_type
        self.model_uuid = model_uuid or str(uuid.uuid4())
        self.saved_files = {}
        self.local_models_dir = os.path.join(os.getcwd(), "local_models", model_type)
        os.makedirs(self.local_models_dir, exist_ok=True)
        
    def save_embedding_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save embedding model locally"""
        filename = f"{self.model_uuid}_embedding.keras"
        filepath = os.path.join(self.local_models_dir, filename)
        
        start_time = time.time()
        model.save(filepath)
        save_time = time.time() - start_time
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        logger.info(f"✅ Local embedding model saved: {filepath} ({file_size:.2f} MB in {save_time:.2f}s)")
        
        self.saved_files['embedding'] = {
            'path': filepath,
            'url': f"file://{filepath}",
            'size_mb': file_size
        }
        return filepath, f"file://{filepath}"
    
    def save_classification_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save classification model locally"""
        filename = f"{self.model_uuid}_classification.keras"
        filepath = os.path.join(self.local_models_dir, filename)
        
        start_time = time.time()
        model.save(filepath)
        save_time = time.time() - start_time
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        logger.info(f"✅ Local classification model saved: {filepath} ({file_size:.2f} MB in {save_time:.2f}s)")
        
        self.saved_files['classification'] = {
            'path': filepath,
            'url': f"file://{filepath}",
            'size_mb': file_size
        }
        return filepath, f"file://{filepath}"
    
    def save_siamese_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save siamese model locally"""
        filename = f"{self.model_uuid}_siamese.keras"
        filepath = os.path.join(self.local_models_dir, filename)
        
        start_time = time.time()
        model.save(filepath)
        save_time = time.time() - start_time
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        logger.info(f"✅ Local siamese model saved: {filepath} ({file_size:.2f} MB in {save_time:.2f}s)")
        
        self.saved_files['siamese'] = {
            'path': filepath,
            'url': f"file://{filepath}",
            'size_mb': file_size
        }
        return filepath, f"file://{filepath}"
    
    def save_mappings(self, student_to_id: Dict, id_to_student: Dict) -> Tuple[str, str]:
        """Save mappings locally"""
        filename = f"{self.model_uuid}_mappings.json"
        filepath = os.path.join(self.local_models_dir, filename)
        
        mappings = {
            'student_to_id': student_to_id,
            'id_to_student': id_to_student,
            'created_at': datetime.utcnow().isoformat(),
            'model_uuid': self.model_uuid
        }
        
        start_time = time.time()
        with open(filepath, 'w') as f:
            json.dump(mappings, f, indent=2)
        save_time = time.time() - start_time
        
        file_size = os.path.getsize(filepath) / 1024  # KB
        logger.info(f"✅ Local mappings saved: {filepath} ({file_size:.2f} KB in {save_time:.2f}s)")
        
        self.saved_files['mappings'] = {
            'path': filepath,
            'url': f"file://{filepath}",
            'size_kb': file_size
        }
        return filepath, f"file://{filepath}"
    
    def get_saved_files(self) -> Dict[str, Dict[str, str]]:
        """Get all saved files with their paths and URLs"""
        return self.saved_files.copy()

def save_signature_models_locally(
    signature_manager: Any,
    model_type: str,
    model_uuid: str = None
) -> Dict[str, Dict[str, str]]:
    """
    Save all signature models locally (FAST - no S3 upload)
    
    Args:
        signature_manager: SignatureEmbeddingModel instance
        model_type: Type of model (individual, global, etc.)
        model_uuid: Optional UUID for the model
    
    Returns:
        Dictionary with saved file information
    """
    saver = LocalModelSaver(model_type, model_uuid)
    
    logger.info(f"🚀 Starting LOCAL save of models (no S3 upload)...")
    start_time = time.time()
    
    try:
        # Save all models locally
        if hasattr(signature_manager, 'embedding_model') and signature_manager.embedding_model:
            saver.save_embedding_model(signature_manager.embedding_model)
        
        if hasattr(signature_manager, 'classification_head') and signature_manager.classification_head:
            saver.save_classification_model(signature_manager.classification_head)
        
        if hasattr(signature_manager, 'siamese_model') and signature_manager.siamese_model:
            saver.save_siamese_model(signature_manager.siamese_model)
        
        if (hasattr(signature_manager, 'student_to_id') and 
            hasattr(signature_manager, 'id_to_student') and
            signature_manager.student_to_id and signature_manager.id_to_student):
            saver.save_mappings(signature_manager.student_to_id, signature_manager.id_to_student)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 All models saved locally in {total_time:.2f}s (INSTANT compared to S3!)")
        
        return saver.get_saved_files()
        
    except Exception as e:
        logger.error(f"❌ Failed to save models locally: {e}")
        raise