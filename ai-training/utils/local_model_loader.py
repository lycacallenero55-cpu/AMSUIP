"""
Local Model Loading Utilities
Load models from local file paths for verification and other operations
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from tensorflow import keras

logger = logging.getLogger(__name__)

def load_model_from_local_path(model_path: str) -> Optional[keras.Model]:
    """
    Load a Keras model from a local file path
    
    Args:
        model_path: Local file path (can be local:// protocol or direct path)
    
    Returns:
        Loaded Keras model or None if failed
    """
    try:
        # Handle local:// protocol
        if model_path.startswith('local://'):
            file_path = model_path.replace('local://', '')
        else:
            file_path = model_path
        
        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Local model file not found: {file_path}")
            return None
        
        # Load the model
        logger.info(f"Loading local model from: {file_path}")
        model = keras.models.load_model(file_path)
        logger.info(f"✅ Successfully loaded local model: {file_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load local model from {model_path}: {e}")
        return None

def load_mappings_from_local_path(mappings_path: str) -> Optional[Dict[str, Any]]:
    """
    Load mappings from a local JSON file
    
    Args:
        mappings_path: Local file path (can be local:// protocol or direct path)
    
    Returns:
        Loaded mappings dictionary or None if failed
    """
    try:
        # Handle local:// protocol
        if mappings_path.startswith('local://'):
            file_path = mappings_path.replace('local://', '')
        else:
            file_path = mappings_path
        
        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Local mappings file not found: {file_path}")
            return None
        
        # Load the mappings
        logger.info(f"Loading local mappings from: {file_path}")
        with open(file_path, 'r') as f:
            mappings = json.load(f)
        logger.info(f"✅ Successfully loaded local mappings: {file_path}")
        return mappings
        
    except Exception as e:
        logger.error(f"Failed to load local mappings from {mappings_path}: {e}")
        return None

def is_local_model_path(model_path: str) -> bool:
    """
    Check if a model path is a local file path
    
    Args:
        model_path: Model path to check
    
    Returns:
        True if it's a local path, False otherwise
    """
    return model_path.startswith('local://') or (not model_path.startswith('http') and not model_path.startswith('s3://'))

def get_local_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a local model file
    
    Args:
        model_path: Local model path
    
    Returns:
        Dictionary with model file information
    """
    try:
        # Handle local:// protocol
        if model_path.startswith('local://'):
            file_path = model_path.replace('local://', '')
        else:
            file_path = model_path
        
        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(file_path):
            return {'exists': False, 'error': 'File not found'}
        
        stat = os.stat(file_path)
        return {
            'exists': True,
            'path': file_path,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified_time': stat.st_mtime,
            'is_file': os.path.isfile(file_path)
        }
        
    except Exception as e:
        return {'exists': False, 'error': str(e)}