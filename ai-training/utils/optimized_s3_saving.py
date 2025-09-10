"""
Optimized S3 Model Saving Utilities
Uses TensorFlow's built-in serialization for maximum efficiency
"""

import io
import json
import uuid
from typing import Dict, Tuple, Optional, Any
import logging
from datetime import datetime

import boto3
from botocore.config import Config as BotoConfig
import tensorflow as tf
from tensorflow import keras
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# S3 client configuration
_session = boto3.session.Session(
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)
_s3 = _session.client(
    "s3",
    config=BotoConfig(
        retries={"max_attempts": 5, "mode": "standard"},
        max_pool_connections=20,
        read_timeout=30,
        connect_timeout=10,
    ),
)

def _resolve_public_base_url() -> str:
    """Get the public base URL for S3 objects"""
    if settings.S3_PUBLIC_BASE_URL:
        return settings.S3_PUBLIC_BASE_URL.rstrip("/")
    return f"https://{settings.S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com"

def _generate_s3_key(model_type: str, model_uuid: str, file_type: str) -> str:
    """Generate S3 key for model files"""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"models/{model_type}/{timestamp}_{model_uuid}_{file_type}.keras"

def _upload_bytes_to_s3(data: bytes, s3_key: str, content_type: str = "application/keras") -> Tuple[str, str]:
    """Upload bytes directly to S3"""
    try:
        acl = "private" if settings.S3_USE_PRESIGNED_GET else "public-read"
        _s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
            ACL=acl
        )
        
        public_url = f"{_resolve_public_base_url()}/{s3_key}"
        logger.info(f"✅ Data uploaded to S3: {s3_key}")
        return s3_key, public_url
        
    except Exception as e:
        logger.error(f"❌ Failed to upload to S3: {e}")
        raise

def _serialize_model_to_bytes(model: keras.Model) -> bytes:
    """Serialize a Keras model to bytes using TensorFlow's built-in serialization"""
    try:
        # Method 1: Try using model.get_config() and model.get_weights()
        # This is more efficient than saving to file
        config = model.get_config()
        weights = model.get_weights()
        
        # Create a serializable representation
        model_data = {
            'config': config,
            'weights': [w.tolist() if hasattr(w, 'tolist') else w for w in weights]
        }
        
        # Serialize to JSON bytes
        json_data = json.dumps(model_data, indent=2)
        return json_data.encode('utf-8')
        
    except Exception as e:
        logger.warning(f"JSON serialization failed, falling back to file method: {e}")
        # Fallback to temporary file method
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            model.save(tmp_path)
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def _deserialize_model_from_bytes(data: bytes) -> keras.Model:
    """Deserialize a Keras model from bytes"""
    try:
        # Try JSON deserialization first
        model_data = json.loads(data.decode('utf-8'))
        config = model_data['config']
        weights = [np.array(w) for w in model_data['weights']]
        
        # Recreate model from config and weights with safe_mode=False for Lambda layers
        model = keras.Model.from_config(config, safe_mode=False)
        model.set_weights(weights)
        return model
        
    except Exception as e:
        logger.warning(f"JSON deserialization failed, falling back to file method: {e}")
        # Fallback to temporary file method
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, 'wb') as f:
                f.write(data)
            return keras.models.load_model(tmp_path, safe_mode=False)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class OptimizedS3ModelSaver:
    """Optimized S3 model saver using TensorFlow's built-in serialization"""
    
    def __init__(self, model_type: str, model_uuid: str = None):
        self.model_type = model_type
        self.model_uuid = model_uuid or str(uuid.uuid4())
        self.uploaded_files = {}
        
    def save_embedding_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save embedding model using optimized serialization"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "embedding")
        model_bytes = _serialize_model_to_bytes(model)
        s3_key, s3_url = _upload_bytes_to_s3(model_bytes, s3_key)
        self.uploaded_files['embedding'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_classification_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save classification model using optimized serialization"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "classification")
        model_bytes = _serialize_model_to_bytes(model)
        s3_key, s3_url = _upload_bytes_to_s3(model_bytes, s3_key)
        self.uploaded_files['classification'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_authenticity_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save authenticity model using optimized serialization"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "authenticity")
        model_bytes = _serialize_model_to_bytes(model)
        s3_key, s3_url = _upload_bytes_to_s3(model_bytes, s3_key)
        self.uploaded_files['authenticity'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_siamese_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save siamese model using optimized serialization"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "siamese")
        model_bytes = _serialize_model_to_bytes(model)
        s3_key, s3_url = _upload_bytes_to_s3(model_bytes, s3_key)
        self.uploaded_files['siamese'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_global_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save global model using optimized serialization"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "global")
        model_bytes = _serialize_model_to_bytes(model)
        s3_key, s3_url = _upload_bytes_to_s3(model_bytes, s3_key)
        self.uploaded_files['global'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_mappings(self, student_to_id: Dict, id_to_student: Dict) -> Tuple[str, str]:
        """Save student mappings directly to S3"""
        mappings_data = {
            'student_to_id': student_to_id,
            'id_to_student': {str(k): v for k, v in id_to_student.items()}
        }
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "mappings").replace('.keras', '.json')
        json_data = json.dumps(mappings_data, indent=2).encode('utf-8')
        s3_key, s3_url = _upload_bytes_to_s3(json_data, s3_key, "application/json")
        self.uploaded_files['mappings'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_centroids(self, centroids: Dict[int, list]) -> Tuple[str, str]:
        """Save centroids data directly to S3"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "centroids").replace('.keras', '.json')
        json_data = json.dumps(centroids, indent=2).encode('utf-8')
        s3_key, s3_url = _upload_bytes_to_s3(json_data, s3_key, "application/json")
        self.uploaded_files['centroids'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def get_uploaded_files(self) -> Dict[str, Dict[str, str]]:
        """Get all uploaded files with their S3 keys and URLs"""
        return self.uploaded_files.copy()

def save_signature_models_optimized(
    signature_manager: Any,
    model_type: str,
    model_uuid: str = None
) -> Dict[str, Dict[str, str]]:
    """
    Save all signature models using optimized serialization
    
    Args:
        signature_manager: SignatureEmbeddingModel instance
        model_type: Type of model (individual, global, etc.)
        model_uuid: Optional UUID for the model
    
    Returns:
        Dictionary with uploaded file information
    """
    saver = OptimizedS3ModelSaver(model_type, model_uuid)
    
    try:
        # Save embedding model if available
        if hasattr(signature_manager, 'embedding_model') and signature_manager.embedding_model:
            saver.save_embedding_model(signature_manager.embedding_model)
        
        # Save classification model if available
        if hasattr(signature_manager, 'classification_head') and signature_manager.classification_head:
            saver.save_classification_model(signature_manager.classification_head)
        
        # Save authenticity model if available
        if hasattr(signature_manager, 'authenticity_head') and signature_manager.authenticity_head:
            saver.save_authenticity_model(signature_manager.authenticity_head)
        
        # Save siamese model if available
        if hasattr(signature_manager, 'siamese_model') and signature_manager.siamese_model:
            saver.save_siamese_model(signature_manager.siamese_model)
        
        # Save mappings if available
        if (hasattr(signature_manager, 'student_to_id') and 
            hasattr(signature_manager, 'id_to_student') and
            signature_manager.student_to_id and signature_manager.id_to_student):
            saver.save_mappings(signature_manager.student_to_id, signature_manager.id_to_student)
        
        logger.info(f"✅ All models saved with optimized serialization for {model_type} model {model_uuid}")
        return saver.get_uploaded_files()
        
    except Exception as e:
        logger.error(f"❌ Failed to save models with optimized serialization: {e}")
        raise