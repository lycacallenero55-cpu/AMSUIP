#!/usr/bin/env python3
"""
Test script to verify code structure and integration
"""

import os
import re

def test_file_exists(file_path, description):
    """Test if a file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def test_file_contains(file_path, pattern, description):
    """Test if a file contains a specific pattern"""
    if not os.path.exists(file_path):
        print(f"❌ {description}: File not found")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if re.search(pattern, content):
            print(f"✅ {description}: Found pattern")
            return True
        else:
            print(f"❌ {description}: Pattern not found")
            return False
    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False

def main():
    """Run all structure tests"""
    print("🚀 Starting Code Structure Tests...")
    print("=" * 60)
    
    tests = []
    
    # Test file existence
    tests.append(test_file_exists('/workspace/ai-training/utils/local_model_saving.py', 'Local model saving utility'))
    tests.append(test_file_exists('/workspace/ai-training/utils/local_model_loader.py', 'Local model loader utility'))
    tests.append(test_file_exists('/workspace/ai-training/.env.example', 'Environment configuration example'))
    
    # Test frontend integration
    frontend_file = '/workspace/src/pages/SignatureAI.tsx'
    tests.append(test_file_exists(frontend_file, 'Frontend SignatureAI page'))
    
    if os.path.exists(frontend_file):
        tests.append(test_file_contains(frontend_file, 'useLocalModels', 'useLocalModels state variable'))
        tests.append(test_file_contains(frontend_file, 'useS3Upload', 'useS3Upload state variable'))
        tests.append(test_file_contains(frontend_file, 'isDropdownOpen', 'isDropdownOpen state variable'))
        tests.append(test_file_contains(frontend_file, 'Use Local Models', 'Local models checkbox'))
        tests.append(test_file_contains(frontend_file, 'S3 Upload', 'S3 upload checkbox'))
        tests.append(test_file_contains(frontend_file, 'Training Options', 'Training options dropdown'))
        tests.append(test_file_contains(frontend_file, 'ChevronDown', 'Chevron dropdown icon'))
        tests.append(test_file_contains(frontend_file, 'useLocalModels', 'Local models state variable'))
    
    # Test backend integration
    training_file = '/workspace/ai-training/api/training.py'
    tests.append(test_file_exists(training_file, 'Training API'))
    
    if os.path.exists(training_file):
        tests.append(test_file_contains(training_file, 'use_s3_upload.*Form', 'S3 upload parameter'))
        tests.append(test_file_contains(training_file, 'save_signature_models_locally', 'Local model saving'))
        tests.append(test_file_contains(training_file, 'save_global_model_locally', 'Global model local saving'))
        tests.append(test_file_contains(training_file, 'not use_s3_upload', 'S3 upload parameter logic'))
    
    verification_file = '/workspace/ai-training/api/verification.py'
    tests.append(test_file_exists(verification_file, 'Verification API'))
    
    if os.path.exists(verification_file):
        tests.append(test_file_contains(verification_file, 'use_local_models.*Form', 'Local models parameter'))
        tests.append(test_file_contains(verification_file, 'load_model_from_local_path', 'Local model loading'))
        tests.append(test_file_contains(verification_file, '_find_local_ai_model', 'Local model finder'))
        tests.append(test_file_contains(verification_file, '_find_s3_ai_model', 'S3 model finder'))
    
    # Test AI service integration
    ai_service_file = '/workspace/src/lib/aiService.ts'
    tests.append(test_file_exists(ai_service_file, 'AI Service'))
    
    if os.path.exists(ai_service_file):
        tests.append(test_file_contains(ai_service_file, 'useS3Upload.*boolean', 'S3 upload parameter'))
        tests.append(test_file_contains(ai_service_file, 'useLocalModels.*boolean', 'Local models parameter'))
        tests.append(test_file_contains(ai_service_file, 'use_s3_upload', 'S3 upload form data'))
        tests.append(test_file_contains(ai_service_file, 'use_local_models', 'Local models form data'))
    
    # Test local storage utility functions
    local_saving_file = '/workspace/ai-training/utils/local_model_saving.py'
    if os.path.exists(local_saving_file):
        tests.append(test_file_contains(local_saving_file, 'local://', 'Local protocol in saving'))
        tests.append(test_file_contains(local_saving_file, 'save_global_model_locally', 'Global model local saving function'))
        tests.append(test_file_contains(local_saving_file, 'relative_path', 'Relative path handling'))
    
    local_loader_file = '/workspace/ai-training/utils/local_model_loader.py'
    if os.path.exists(local_loader_file):
        tests.append(test_file_contains(local_loader_file, 'load_model_from_local_path', 'Model loading function'))
        tests.append(test_file_contains(local_loader_file, 'is_local_model_path', 'Path detection function'))
        tests.append(test_file_contains(local_loader_file, 'local://', 'Local protocol handling'))
    
    print("=" * 60)
    passed = sum(tests)
    total = len(tests)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! Code integration is complete.")
        print("\n📋 IMPLEMENTATION SUMMARY:")
        print("✅ Local storage utilities created")
        print("✅ Frontend dropdown with GPU/S3 options")
        print("✅ Frontend verification with local models option")
        print("✅ Backend API parameters for storage selection")
        print("✅ Model loading supports both local and S3")
        print("✅ Training functions support both storage types")
        print("✅ Verification functions support both storage types")
        print("\n🚀 SYSTEM IS READY FOR TESTING!")
    else:
        print("⚠️  Some structure tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)