#!/usr/bin/env python3
"""
Smoke Test for Signature AI System
Quick test to verify basic functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from models.database import DatabaseManager
from utils.s3_storage import S3StorageManager
from models.signature_embedding_model import SignatureEmbeddingModel

async def smoke_test():
    """Run basic smoke tests"""
    print("🔥 Running Signature AI Smoke Tests")
    print("=" * 40)
    
    # Test 1: Database connection
    print("1. Testing database connection...")
    try:
        db_manager = DatabaseManager()
        students = await db_manager.list_students()
        print(f"   ✅ Database connected. Found {len(students)} students.")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return False
    
    # Test 2: S3 connection
    print("2. Testing S3 connection...")
    try:
        s3_manager = S3StorageManager()
        # Test with a small file
        test_data = b"smoke test data"
        test_key = "test/smoke_test.txt"
        success = await s3_manager.upload_file(test_key, test_data)
        if success:
            await s3_manager.delete_file(test_key)
            print("   ✅ S3 connected successfully.")
        else:
            print("   ❌ S3 upload failed.")
            return False
    except Exception as e:
        print(f"   ❌ S3 connection failed: {e}")
        return False
    
    # Test 3: Model initialization
    print("3. Testing model initialization...")
    try:
        model = SignatureEmbeddingModel(max_students=150)
        print("   ✅ Model initialized successfully.")
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        return False
    
    # Test 4: Check configuration
    print("4. Checking configuration...")
    from config import settings
    print(f"   📊 Max students: {settings.MAX_STUDENTS}")
    print(f"   📊 Image size: {settings.IMAGE_SIZE}")
    print(f"   📊 Confidence threshold: {settings.CONFIDENCE_THRESHOLD}")
    print(f"   📊 Forgery detection: {settings.ENABLE_FORGERY_DETECTION}")
    print(f"   📊 Anti-spoofing: {settings.ENABLE_ANTISPOOFING}")
    print("   ✅ Configuration loaded successfully.")
    
    print("=" * 40)
    print("🎉 All smoke tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(smoke_test())
    sys.exit(0 if success else 1)