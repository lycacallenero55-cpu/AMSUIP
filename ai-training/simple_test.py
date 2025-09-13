#!/usr/bin/env python3
"""
Simple smoke test for the AI signature verification system
"""
import sys
import os
import tempfile
import numpy as np
from PIL import Image
import io

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image():
    """Create a simple test signature image"""
    img = Image.new('RGB', (200, 100), color='white')
    pixels = np.array(img)
    # Add some noise to simulate signature strokes
    noise = np.random.randint(0, 50, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255)
    img = Image.fromarray(pixels.astype('uint8'))
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ PIL imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PIL: {e}")
        return False
    
    try:
        from models.signature_embedding_model import SignatureEmbeddingModel
        print("✓ SignatureEmbeddingModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SignatureEmbeddingModel: {e}")
        return False
    
    try:
        from models.global_signature_model import GlobalSignatureModel
        print("✓ GlobalSignatureModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GlobalSignatureModel: {e}")
        return False
    
    try:
        from utils.s3_storage import S3Storage
        print("✓ S3Storage imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import S3Storage: {e}")
        return False
    
    try:
        from models.database import DatabaseManager
        print("✓ DatabaseManager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DatabaseManager: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that models can be created"""
    print("\nTesting model creation...")
    
    try:
        from models.signature_embedding_model import SignatureEmbeddingModel
        model = SignatureEmbeddingModel()
        print("✓ SignatureEmbeddingModel created successfully")
    except Exception as e:
        print(f"✗ Failed to create SignatureEmbeddingModel: {e}")
        return False
    
    try:
        from models.global_signature_model import GlobalSignatureModel
        global_model = GlobalSignatureModel()
        print("✓ GlobalSignatureModel created successfully")
    except Exception as e:
        print(f"✗ Failed to create GlobalSignatureModel: {e}")
        return False
    
    return True

def test_data_preparation():
    """Test data preparation functionality"""
    print("\nTesting data preparation...")
    
    try:
        from models.signature_embedding_model import SignatureEmbeddingModel
        model = SignatureEmbeddingModel()
        
        # Create test images
        test_images = [create_test_image() for _ in range(5)]
        
        # Test data preparation
        training_data = model.prepare_training_data(
            genuine_images=test_images[:3],
            forged_images=test_images[3:],
            student_id=1
        )
        
        if training_data is None:
            print("✗ Data preparation returned None")
            return False
        
        if len(training_data['genuine']) != 3:
            print(f"✗ Expected 3 genuine images, got {len(training_data['genuine'])}")
            return False
        
        if len(training_data['forged']) != 2:
            print(f"✗ Expected 2 forged images, got {len(training_data['forged'])}")
            return False
        
        print("✓ Data preparation successful")
        return True
        
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

def test_model_training():
    """Test model training with minimal data"""
    print("\nTesting model training...")
    
    try:
        from models.signature_embedding_model import SignatureEmbeddingModel
        model = SignatureEmbeddingModel()
        
        # Create test images
        test_images = [create_test_image() for _ in range(5)]
        
        # Prepare training data
        training_data = model.prepare_training_data(
            genuine_images=test_images[:3],
            forged_images=test_images[3:],
            student_id=1
        )
        
        # Mock TensorFlow callbacks to avoid early stopping
        import unittest.mock
        with unittest.mock.patch('models.signature_embedding_model.tf.keras.callbacks.EarlyStopping') as mock_early_stopping:
            mock_early_stopping.return_value = unittest.mock.MagicMock()
            
            # Train model with minimal epochs
            result = model.train_model(
                training_data=training_data,
                epochs=1,  # Minimal training
                batch_size=1,
                learning_rate=0.001
            )
        
        if not result['success']:
            print(f"✗ Model training failed: {result.get('error', 'Unknown error')}")
            return False
        
        if 'accuracy' not in result:
            print("✗ Model training result missing accuracy")
            return False
        
        if 'loss' not in result:
            print("✗ Model training result missing loss")
            return False
        
        print(f"✓ Model training successful - Accuracy: {result['accuracy']:.4f}, Loss: {result['loss']:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False

def test_verification():
    """Test signature verification"""
    print("\nTesting signature verification...")
    
    try:
        from models.signature_embedding_model import SignatureEmbeddingModel
        model = SignatureEmbeddingModel()
        
        # Create test images
        test_images = [create_test_image() for _ in range(5)]
        
        # Prepare training data
        training_data = model.prepare_training_data(
            genuine_images=test_images[:3],
            forged_images=test_images[3:],
            student_id=1
        )
        
        # Train model
        import unittest.mock
        with unittest.mock.patch('models.signature_embedding_model.tf.keras.callbacks.EarlyStopping') as mock_early_stopping:
            mock_early_stopping.return_value = unittest.mock.MagicMock()
            
            train_result = model.train_model(
                training_data=training_data,
                epochs=1,
                batch_size=1,
                learning_rate=0.001
            )
        
        if not train_result['success']:
            print("✗ Model training failed, cannot test verification")
            return False
        
        # Test verification
        test_image = test_images[0]
        verification_result = model.verify_signature(
            test_image=test_image,
            student_id=1,
            threshold=0.5
        )
        
        if 'is_match' not in verification_result:
            print("✗ Verification result missing is_match")
            return False
        
        if 'confidence' not in verification_result:
            print("✗ Verification result missing confidence")
            return False
        
        if 'is_genuine' not in verification_result:
            print("✗ Verification result missing is_genuine")
            return False
        
        # Check that forgery detection is disabled
        if not verification_result['is_genuine']:
            print("✗ Forgery detection should be disabled (is_genuine should be True)")
            return False
        
        print(f"✓ Signature verification successful - Match: {verification_result['is_match']}, Confidence: {verification_result['confidence']:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Signature verification failed: {e}")
        return False

def test_configuration():
    """Test configuration settings"""
    print("\nTesting configuration...")
    
    try:
        from config import ENABLE_FORGERY_DETECTION
        
        if ENABLE_FORGERY_DETECTION:
            print("✗ Forgery detection should be disabled")
            return False
        
        print("✓ Forgery detection is properly disabled")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all smoke tests"""
    print("AI Signature Verification System - Smoke Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Data Preparation Test", test_data_preparation),
        ("Model Training Test", test_model_training),
        ("Verification Test", test_verification),
        ("Configuration Test", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"SMOKE TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SMOKE TESTS PASSED! The AI system is working correctly.")
        return True
    else:
        print("❌ Some smoke tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)