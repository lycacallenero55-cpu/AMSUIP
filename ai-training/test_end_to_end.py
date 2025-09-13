#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Signature AI System
Tests the complete pipeline: upload → train → verify → return correct owner or "no_match"
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from models.database import DatabaseManager
from utils.s3_storage import S3StorageManager
from models.signature_embedding_model import SignatureEmbeddingModel

class SignatureAITester:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.s3_manager = S3StorageManager()
        self.base_url = "http://localhost:8000"
        self.test_results = []
        
    def create_test_signature(self, student_name: str, style: str = "normal") -> bytes:
        """Create a synthetic signature for testing"""
        # Create a white background
        img = Image.new('RGB', (400, 200), 'white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw signature based on style
        if style == "cursive":
            # Draw cursive-like signature
            points = [(50, 100), (100, 80), (150, 90), (200, 85), (250, 95), (300, 88), (350, 92)]
            for i in range(len(points) - 1):
                draw.line([points[i], points[i+1]], fill='black', width=3)
        elif style == "block":
            # Draw block letters
            draw.text((50, 80), student_name[:10], fill='black', font=font)
        else:
            # Draw normal signature
            draw.text((50, 80), student_name, fill='black', font=font)
            # Add some flourishes
            draw.line([(50, 120), (200, 120)], fill='black', width=2)
            draw.line([(200, 120), (250, 100)], fill='black', width=2)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    def create_test_students(self) -> List[Dict]:
        """Create test students with synthetic signatures"""
        students = []
        test_names = ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince"]
        
        for i, name in enumerate(test_names):
            student_id = i + 1
            students.append({
                "id": student_id,
                "name": name,
                "email": f"{name.lower().replace(' ', '.')}@test.com",
                "signatures": []
            })
            
            # Create 3-5 signatures per student with different styles
            styles = ["normal", "cursive", "block"]
            for j in range(3):
                signature_data = self.create_test_signature(name, styles[j % len(styles)])
                students[-1]["signatures"].append({
                    "data": signature_data,
                    "filename": f"{name}_{j+1}.png"
                })
        
        return students
    
    async def test_database_connection(self) -> bool:
        """Test database connectivity"""
        try:
            print("🔍 Testing database connection...")
            students = await self.db_manager.list_students()
            print(f"✅ Database connected. Found {len(students)} existing students.")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    async def test_s3_connection(self) -> bool:
        """Test S3 connectivity"""
        try:
            print("🔍 Testing S3 connection...")
            # Test S3 by uploading a small test file
            test_data = b"test data"
            test_key = "test/connection_test.txt"
            success = await self.s3_manager.upload_file(test_key, test_data)
            if success:
                # Clean up test file
                await self.s3_manager.delete_file(test_key)
                print("✅ S3 connected successfully.")
                return True
            else:
                print("❌ S3 upload failed.")
                return False
        except Exception as e:
            print(f"❌ S3 connection failed: {e}")
            return False
    
    async def test_upload_signatures(self, students: List[Dict]) -> bool:
        """Test uploading signatures for students"""
        try:
            print("🔍 Testing signature upload...")
            
            for student in students:
                print(f"  📝 Uploading signatures for {student['name']}...")
                
                for i, signature in enumerate(student["signatures"]):
                    # Upload signature to S3
                    s3_key = f"signatures/{student['id']}/{signature['filename']}"
                    success = await self.s3_manager.upload_file(s3_key, signature["data"])
                    
                    if not success:
                        print(f"❌ Failed to upload signature {i+1} for {student['name']}")
                        return False
                    
                    # Create database record
                    signature_record = {
                        "student_id": student["id"],
                        "s3_key": s3_key,
                        "filename": signature["filename"],
                        "is_genuine": True,
                        "upload_timestamp": "now()"
                    }
                    
                    success = await self.db_manager.create_signature(signature_record)
                    if not success:
                        print(f"❌ Failed to create database record for signature {i+1}")
                        return False
                
                print(f"  ✅ Uploaded {len(student['signatures'])} signatures for {student['name']}")
            
            print("✅ All signatures uploaded successfully.")
            return True
            
        except Exception as e:
            print(f"❌ Signature upload failed: {e}")
            return False
    
    async def test_training_pipeline(self) -> bool:
        """Test the AI training pipeline"""
        try:
            print("🔍 Testing AI training pipeline...")
            
            # Start training via API
            training_url = f"{self.base_url}/api/training/start-global"
            response = requests.post(training_url, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Training API failed: {response.status_code} - {response.text}")
                return False
            
            training_data = response.json()
            job_id = training_data.get("job_id")
            
            if not job_id:
                print("❌ No job ID returned from training API")
                return False
            
            print(f"  🚀 Training started with job ID: {job_id}")
            
            # Monitor training progress
            progress_url = f"{self.base_url}/api/training/progress/{job_id}"
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                try:
                    progress_response = requests.get(progress_url, timeout=10)
                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        status = progress_data.get("status", "unknown")
                        progress = progress_data.get("progress", 0)
                        
                        print(f"  📊 Training progress: {progress}% - Status: {status}")
                        
                        if status == "completed":
                            print("✅ Training completed successfully!")
                            return True
                        elif status == "failed":
                            error = progress_data.get("error", "Unknown error")
                            print(f"❌ Training failed: {error}")
                            return False
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    print(f"⚠️  Progress check failed: {e}")
                    time.sleep(5)
            
            print("❌ Training timed out")
            return False
            
        except Exception as e:
            print(f"❌ Training pipeline test failed: {e}")
            return False
    
    async def test_verification_accuracy(self, students: List[Dict]) -> bool:
        """Test verification accuracy with known signatures"""
        try:
            print("🔍 Testing verification accuracy...")
            
            correct_predictions = 0
            total_tests = 0
            
            for student in students:
                print(f"  🎯 Testing verification for {student['name']}...")
                
                # Test with each signature
                for i, signature in enumerate(student["signatures"]):
                    # Create a test file
                    test_file = io.BytesIO(signature["data"])
                    files = {"test_file": ("test.png", test_file, "image/png")}
                    
                    # Call verification API
                    verify_url = f"{self.base_url}/api/verify"
                    response = requests.post(verify_url, files=files, timeout=30)
                    
                    if response.status_code != 200:
                        print(f"    ❌ Verification failed for signature {i+1}: {response.status_code}")
                        continue
                    
                    result = response.json()
                    predicted_name = result.get("predicted_student_name", "")
                    is_match = result.get("is_match", False)
                    confidence = result.get("confidence", 0.0)
                    
                    print(f"    📊 Signature {i+1}: Predicted={predicted_name}, Match={is_match}, Confidence={confidence:.3f}")
                    
                    total_tests += 1
                    
                    # Check if prediction is correct
                    if is_match and predicted_name == student["name"]:
                        correct_predictions += 1
                        print(f"    ✅ Correct prediction!")
                    else:
                        print(f"    ❌ Incorrect prediction (expected: {student['name']})")
            
            accuracy = correct_predictions / total_tests if total_tests > 0 else 0
            print(f"📊 Overall accuracy: {accuracy:.2%} ({correct_predictions}/{total_tests})")
            
            # Require at least 60% accuracy for small datasets
            if accuracy >= 0.6:
                print("✅ Verification accuracy test passed!")
                return True
            else:
                print("❌ Verification accuracy too low")
                return False
                
        except Exception as e:
            print(f"❌ Verification accuracy test failed: {e}")
            return False
    
    async def test_unknown_signature_detection(self) -> bool:
        """Test detection of unknown signatures"""
        try:
            print("🔍 Testing unknown signature detection...")
            
            # Create a signature from an untrained student
            unknown_signature = self.create_test_signature("Unknown Student", "cursive")
            test_file = io.BytesIO(unknown_signature)
            files = {"test_file": ("unknown.png", test_file, "image/png")}
            
            # Call verification API
            verify_url = f"{self.base_url}/api/verify"
            response = requests.post(verify_url, files=files, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Verification failed for unknown signature: {response.status_code}")
                return False
            
            result = response.json()
            is_match = result.get("is_match", True)  # Should be False
            confidence = result.get("confidence", 0.0)
            
            print(f"📊 Unknown signature: Match={is_match}, Confidence={confidence:.3f}")
            
            # Should return no_match for unknown signatures
            if not is_match:
                print("✅ Unknown signature correctly identified as no_match!")
                return True
            else:
                print("❌ Unknown signature incorrectly identified as match")
                return False
                
        except Exception as e:
            print(f"❌ Unknown signature detection test failed: {e}")
            return False
    
    async def cleanup_test_data(self, students: List[Dict]):
        """Clean up test data"""
        try:
            print("🧹 Cleaning up test data...")
            
            for student in students:
                # Delete signatures from S3
                for signature in student["signatures"]:
                    s3_key = f"signatures/{student['id']}/{signature['filename']}"
                    await self.s3_manager.delete_file(s3_key)
                
                # Delete signature records from database
                await self.db_manager.delete_student_signatures(student["id"])
            
            print("✅ Test data cleaned up.")
            
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")
    
    async def run_full_test(self):
        """Run the complete end-to-end test"""
        print("🚀 Starting Signature AI End-to-End Test")
        print("=" * 50)
        
        # Test infrastructure
        if not await self.test_database_connection():
            return False
        
        if not await self.test_s3_connection():
            return False
        
        # Create test data
        students = self.create_test_students()
        print(f"📝 Created {len(students)} test students with signatures")
        
        try:
            # Test upload
            if not await self.test_upload_signatures(students):
                return False
            
            # Test training
            if not await self.test_training_pipeline():
                return False
            
            # Wait a bit for model to be fully loaded
            print("⏳ Waiting for model to be fully loaded...")
            await asyncio.sleep(10)
            
            # Test verification accuracy
            if not await self.test_verification_accuracy(students):
                return False
            
            # Test unknown signature detection
            if not await self.test_unknown_signature_detection():
                return False
            
            print("=" * 50)
            print("🎉 ALL TESTS PASSED! Signature AI system is working correctly.")
            return True
            
        finally:
            # Always cleanup
            await self.cleanup_test_data(students)

async def main():
    """Main test function"""
    tester = SignatureAITester()
    success = await tester.run_full_test()
    
    if success:
        print("\n✅ End-to-end test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ End-to-end test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())