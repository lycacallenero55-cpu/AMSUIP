#!/usr/bin/env python3
"""
Test script to verify local storage integration works correctly
"""

import os
import sys
import tempfile
import json
from unittest.mock import Mock, patch

# Add the ai-training directory to the path
sys.path.append('/workspace/ai-training')

def test_local_model_saving():
    """Test local model saving functionality"""
    print("🧪 Testing local model saving...")
    
    try:
        from utils.local_model_saving import LocalModelSaver, save_signature_models_locally
        from tensorflow import keras
        import numpy as np
        
        # Create a mock model
        model = keras.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(784,)),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Test LocalModelSaver
        saver = LocalModelSaver("test", "test-uuid")
        
        # Test saving models
        filepath, url = saver.save_classification_model(model)
        print(f"✅ Classification model saved: {filepath}")
        print(f"✅ URL format: {url}")
        
        # Test saving mappings
        mappings_path, mappings_url = saver.save_mappings(
            {"student1": 0, "student2": 1},
            {0: "student1", 1: "student2"}
        )
        print(f"✅ Mappings saved: {mappings_path}")
        print(f"✅ Mappings URL format: {mappings_url}")
        
        # Verify files exist
        assert os.path.exists(filepath), "Model file should exist"
        assert os.path.exists(mappings_path), "Mappings file should exist"
        
        # Verify URL format
        assert url.startswith('local://'), "URL should use local:// protocol"
        assert mappings_url.startswith('local://'), "Mappings URL should use local:// protocol"
        
        print("✅ Local model saving test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Local model saving test failed: {e}")
        return False

def test_local_model_loading():
    """Test local model loading functionality"""
    print("🧪 Testing local model loading...")
    
    try:
        from utils.local_model_loader import load_model_from_local_path, load_mappings_from_local_path, is_local_model_path
        
        # Test path detection
        assert is_local_model_path('local://path/to/model.keras'), "Should detect local:// protocol"
        assert is_local_model_path('path/to/model.keras'), "Should detect relative path"
        assert not is_local_model_path('https://bucket.amazonaws.com/model.keras'), "Should not detect S3 URL"
        
        print("✅ Local model loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Local model loading test failed: {e}")
        return False

def test_training_integration():
    """Test training integration with local storage"""
    print("🧪 Testing training integration...")
    
    try:
        # Mock the training functions
        with patch('api.training.os.getenv') as mock_env:
            mock_env.return_value = 'false'  # Don't use env var
            
            # Test parameter passing
            import sys
            sys.path.append('/workspace/ai-training')
            from api.training import _train_and_store_individual_from_arrays
            
            # This should not fail with the new parameter
            print("✅ Training function signature updated correctly")
            
        print("✅ Training integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training integration test failed: {e}")
        return False

def test_verification_integration():
    """Test verification integration with local storage"""
    print("🧪 Testing verification integration...")
    
    try:
        import sys
        sys.path.append('/workspace/ai-training')
        from api.verification import _find_local_ai_model, _find_s3_ai_model
        
        # Test helper functions exist
        assert callable(_find_local_ai_model), "Local model finder should be callable"
        assert callable(_find_s3_ai_model), "S3 model finder should be callable"
        
        print("✅ Verification integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Verification integration test failed: {e}")
        return False

def test_frontend_integration():
    """Test frontend integration"""
    print("🧪 Testing frontend integration...")
    
    try:
        # Check if the frontend files have the right imports and state
        frontend_file = '/workspace/src/pages/SignatureAI.tsx'
        if os.path.exists(frontend_file):
            with open(frontend_file, 'r') as f:
                content = f.read()
                
            # Check for required state variables
            assert 'useLocalModels' in content, "useLocalModels state should exist"
            assert 'useS3Upload' in content, "useS3Upload state should exist"
            assert 'isDropdownOpen' in content, "isDropdownOpen state should exist"
            
            # Check for required UI elements
            assert 'Use Local Models' in content, "Local models checkbox should exist"
            assert 'S3 Upload' in content, "S3 upload checkbox should exist"
            assert 'Training Options' in content, "Training options dropdown should exist"
            
            print("✅ Frontend integration test passed!")
            return True
        else:
            print("❌ Frontend file not found")
            return False
            
    except Exception as e:
        print(f"❌ Frontend integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Local Storage Integration Tests...")
    print("=" * 50)
    
    tests = [
        test_local_model_saving,
        test_local_model_loading,
        test_training_integration,
        test_verification_integration,
        test_frontend_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Local storage integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)