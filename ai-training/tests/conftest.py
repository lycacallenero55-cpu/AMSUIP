"""
Test configuration and fixtures
"""
import pytest
import os
import sys
from unittest.mock import Mock, AsyncMock

# Add the ai_training directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set test environment variables
os.environ['ENVIRONMENT'] = 'test'
os.environ['S3_BUCKET_NAME'] = 'test-bucket'
os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
os.environ['SUPABASE_KEY'] = 'test-key'


@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        'S3_BUCKET_NAME': 'test-bucket',
        'SUPABASE_URL': 'https://test.supabase.co',
        'SUPABASE_KEY': 'test-key',
        'ENABLE_FORGERY_DETECTION': False,
        'MIN_GENUINE_SAMPLES': 3,
        'CONFIDENCE_THRESHOLD': 0.6,
        'MODEL_SAVE_PATH': '/tmp/test_models'
    }


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing"""
    client = Mock()
    client.upload_file = Mock()
    client.download_file = Mock()
    client.list_objects_v2 = Mock()
    client.head_object = Mock()
    client.delete_object = Mock()
    return client


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing"""
    client = Mock()
    client.table = Mock()
    return client


@pytest.fixture
def sample_signature_data():
    """Generate sample signature data for testing"""
    import numpy as np
    
    # Create sample signature images
    images = []
    for i in range(5):
        # Create a simple signature pattern
        img = np.random.rand(28, 28, 1) * 255
        # Add some structure to make it look like a signature
        img[10:18, 5:23] = 0  # Horizontal line
        img[5:13, 8:16] = 0   # Vertical line
        images.append(img.astype(np.uint8))
    
    return {
        'images': images,
        'student_ids': [1, 1, 1, 1, 1],
        'labels': [1, 1, 1, 1, 1]
    }


@pytest.fixture
def mock_model():
    """Mock AI model for testing"""
    model = Mock()
    model.compile_embedding_model = Mock()
    model.compile_classification_model = Mock()
    model.train_classification_only = Mock()
    model.verify_signature = Mock()
    model.identify_signature_owner = Mock()
    model.load_classification_model = Mock()
    return model