"""
Direct S3 Model Saving Utilities
Eliminates the two-step process of saving locally then uploading to S3
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
        read_timeout=30,  # Increased for model uploads
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

def _upload_model_to_s3(model: keras.Model, s3_key: str) -> Tuple[str, str]:
    """Upload a Keras model directly to S3 with minimal local file usage"""
    try:
        # TensorFlow model.save() requires a file path, not BytesIO
        # Use temporary file approach for compatibility
        import tempfile
        import os
        
        # Create temporary file with proper extension
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save model to temporary file
            model.save(tmp_path)
            
            # Read the temporary file
            with open(tmp_path, 'rb') as f:
                model_data = f.read()
            
        finally:
            # Always clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Upload to S3
        acl = "private" if settings.S3_USE_PRESIGNED_GET else "public-read"
        _s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=s3_key,
            Body=model_data,
            ContentType="application/keras",
            ACL=acl
        )
        
        # Generate public URL
        public_url = f"{_resolve_public_base_url()}/{s3_key}"
        
        logger.info(f"✅ Model uploaded directly to S3: {s3_key}")
        return s3_key, public_url
        
    except Exception as e:
        logger.error(f"❌ Failed to upload model to S3: {e}")
        raise

def _upload_json_to_s3(data: Dict[str, Any], s3_key: str) -> Tuple[str, str]:
    """Upload JSON data directly to S3"""
    try:
        json_data = json.dumps(data, indent=2).encode('utf-8')
        
        acl = "private" if settings.S3_USE_PRESIGNED_GET else "public-read"
        _s3.put_object(
            Bucket=settings.S3_BUCKET,
            Key=s3_key,
            Body=json_data,
            ContentType="application/json",
            ACL=acl
        )
        
        public_url = f"{_resolve_public_base_url()}/{s3_key}"
        logger.info(f"✅ JSON data uploaded directly to S3: {s3_key}")
        return s3_key, public_url
        
    except Exception as e:
        logger.error(f"❌ Failed to upload JSON to S3: {e}")
        raise

class DirectS3ModelSaver:
    """Direct S3 model saver that eliminates local file operations"""
    
    def __init__(self, model_type: str, model_uuid: str = None):
        self.model_type = model_type
        self.model_uuid = model_uuid or str(uuid.uuid4())
        self.uploaded_files = {}
        
    def save_embedding_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save embedding model directly to S3"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "embedding")
        s3_key, s3_url = _upload_model_to_s3(model, s3_key)
        self.uploaded_files['embedding'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_classification_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save classification model directly to S3"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "classification")
        s3_key, s3_url = _upload_model_to_s3(model, s3_key)
        self.uploaded_files['classification'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_authenticity_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save authenticity model directly to S3"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "authenticity")
        s3_key, s3_url = _upload_model_to_s3(model, s3_key)
        self.uploaded_files['authenticity'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_siamese_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save siamese model directly to S3"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "siamese")
        s3_key, s3_url = _upload_model_to_s3(model, s3_key)
        self.uploaded_files['siamese'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_global_model(self, model: keras.Model) -> Tuple[str, str]:
        """Save global model directly to S3"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "global")
        s3_key, s3_url = _upload_model_to_s3(model, s3_key)
        self.uploaded_files['global'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_mappings(self, student_to_id: Dict, id_to_student: Dict) -> Tuple[str, str]:
        """Save student mappings directly to S3"""
        mappings_data = {
            'student_to_id': student_to_id,
            'id_to_student': {str(k): v for k, v in id_to_student.items()}
        }
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "mappings").replace('.keras', '.json')
        s3_key, s3_url = _upload_json_to_s3(mappings_data, s3_key)
        self.uploaded_files['mappings'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def save_centroids(self, centroids: Dict[int, list]) -> Tuple[str, str]:
        """Save centroids data directly to S3"""
        s3_key = _generate_s3_key(self.model_type, self.model_uuid, "centroids").replace('.keras', '.json')
        s3_key, s3_url = _upload_json_to_s3(centroids, s3_key)
        self.uploaded_files['centroids'] = {'key': s3_key, 'url': s3_url}
        return s3_key, s3_url
    
    def get_uploaded_files(self) -> Dict[str, Dict[str, str]]:
        """Get all uploaded files with their S3 keys and URLs"""
        return self.uploaded_files.copy()

def save_signature_models_directly(
    signature_manager: Any,
    model_type: str,
    model_uuid: str = None
) -> Dict[str, Dict[str, str]]:
    """
    Save all signature models directly to S3 without local files
    
    Args:
        signature_manager: SignatureEmbeddingModel instance
        model_type: Type of model (individual, global, etc.)
        model_uuid: Optional UUID for the model
    
    Returns:
        Dictionary with uploaded file information
    """
    saver = DirectS3ModelSaver(model_type, model_uuid)
    
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
        
        logger.info(f"✅ All models saved directly to S3 for {model_type} model {model_uuid}")
        return saver.get_uploaded_files()
        
    except Exception as e:
        logger.error(f"❌ Failed to save models directly to S3: {e}")
        raise

def save_global_model_directly(
    global_model: Any,
    model_type: str = "global",
    model_uuid: str = None
) -> Tuple[str, str]:
    """
    Save global model directly to S3
    
    Args:
        global_model: GlobalSignatureVerificationModel instance
        model_type: Type of model
        model_uuid: Optional UUID for the model
    
    Returns:
        Tuple of (s3_key, s3_url)
    """
    saver = DirectS3ModelSaver(model_type, model_uuid)
    
    try:
        # Save the main global model
        if hasattr(global_model, 'embedding_model') and global_model.embedding_model:
            s3_key, s3_url = saver.save_embedding_model(global_model.embedding_model)
        else:
            # Fallback: save the model itself if it's a Keras model
            s3_key, s3_url = saver.save_global_model(global_model)
        
        logger.info(f"✅ Global model saved directly to S3: {s3_key}")
        return s3_key, s3_url
        
    except Exception as e:
        logger.error(f"❌ Failed to save global model directly to S3: {e}")
        raise