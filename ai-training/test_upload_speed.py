#!/usr/bin/env python3
"""
Upload Speed Test Script
Test different upload methods to find the fastest one
"""

import os
import time
import tempfile
import numpy as np
from tensorflow import keras
from utils.optimized_s3_saving import save_signature_models_optimized
from utils.local_model_saving import save_signature_models_locally
from models.signature_embedding_model import SignatureEmbeddingModel

def create_test_model():
    """Create a test model similar to your signature models"""
    # Create a simple test model
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(224, 224, 3)),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_test_manager():
    """Create a test SignatureEmbeddingModel manager"""
    manager = SignatureEmbeddingModel(max_students=150)
    
    # Add test models
    manager.embedding_model = create_test_model()
    manager.classification_head = create_test_model()
    manager.siamese_model = create_test_model()
    
    # Add test mappings
    manager.student_to_id = {'Student1': 0, 'Student2': 1}
    manager.id_to_student = {0: 'Student1', 1: 'Student2'}
    
    return manager

def test_local_storage():
    """Test local storage speed"""
    print("🚀 Testing LOCAL storage...")
    manager = create_test_manager()
    
    start_time = time.time()
    try:
        files = save_signature_models_locally(manager, "test", "speed_test_local")
        end_time = time.time()
        
        total_time = end_time - start_time
        print(f"✅ Local storage completed in {total_time:.2f}s")
        print(f"📁 Files saved: {list(files.keys())}")
        return total_time
    except Exception as e:
        print(f"❌ Local storage failed: {e}")
        return None

def test_s3_parallel():
    """Test S3 parallel upload speed"""
    print("☁️ Testing S3 parallel uploads...")
    manager = create_test_manager()
    
    start_time = time.time()
    try:
        files = save_signature_models_optimized(manager, "test", "speed_test_s3")
        end_time = time.time()
        
        total_time = end_time - start_time
        print(f"✅ S3 parallel upload completed in {total_time:.2f}s")
        print(f"☁️ Files uploaded: {list(files.keys())}")
        return total_time
    except Exception as e:
        print(f"❌ S3 parallel upload failed: {e}")
        return None

def main():
    """Run speed tests"""
    print("🏁 Starting Upload Speed Tests")
    print("=" * 50)
    
    # Test local storage
    local_time = test_local_storage()
    
    print("\n" + "=" * 50)
    
    # Test S3 parallel uploads
    s3_time = test_s3_parallel()
    
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY:")
    print("=" * 50)
    
    if local_time:
        print(f"🚀 Local Storage: {local_time:.2f}s")
    
    if s3_time:
        print(f"☁️ S3 Parallel: {s3_time:.2f}s")
    
    if local_time and s3_time:
        speedup = s3_time / local_time
        print(f"⚡ Local is {speedup:.1f}x faster than S3")
        
        if speedup > 10:
            print("💡 RECOMMENDATION: Use local storage for development!")
        elif speedup > 3:
            print("💡 RECOMMENDATION: Local storage is significantly faster")
        else:
            print("💡 RECOMMENDATION: S3 parallel uploads are reasonably fast")
    
    print("\n🔧 To use local storage, set: USE_LOCAL_STORAGE=true")
    print("🔧 To use S3 parallel, set: USE_LOCAL_STORAGE=false")

if __name__ == "__main__":
    main()