#!/usr/bin/env python3
"""
Fast Training Test Script
Run this to test training with minimal setup and fast execution
"""

import boto3
import os
from config import settings
import asyncio
import json
import base64
import time

async def run_fast_training_test():
    """Run a fast training test to check if everything works"""

    print("=" * 60)
    print("FAST TRAINING TEST")
    print("=" * 60)
    print("This will test the training pipeline with minimal data")
    print()

    # Create minimal test data
    test_data = {
        "test_student": {
            "genuine": [
                # Create a simple 224x224x3 array (minimal valid image data)
                {
                    "array": [[[[0.5] * 224] * 224] * 3],
                    "shape": [224, 224, 3]
                }
            ],
            "forged": []
        }
    }

    # Upload test data to S3
    s3_client = boto3.client('s3', region_name=settings.AWS_REGION)
    test_key = f'test_training_data/test_{int(time.time())}.json'

    try:
        s3_client.put_object(
            Bucket=settings.S3_BUCKET,
            Key=test_key,
            Body=json.dumps(test_data),
            ContentType='application/json'
        )
        print(f"✅ Test data uploaded to S3: {test_key}")
    except Exception as e:
        print(f"❌ Failed to upload test data: {e}")
        return

    # Try to run training
    from utils.aws_gpu_training import gpu_training_manager

    try:
        print("✅ GPU manager available")
        print(f"✅ Using instance: {settings.EXISTING_GPU_INSTANCE_ID}")

        # Start training
        print("🚀 Starting fast training test...")
        result = await gpu_training_manager.start_gpu_training(
            test_data, "test_job_123", 999
        )

        if result.get('success'):
            print("✅ Training completed successfully!")
            print(f"   Result: {result}")
        else:
            print(f"❌ Training failed: {result}")

    except Exception as e:
        print(f"❌ Training error: {e}")
        print("This might indicate:")
        print("- EC2 instance not running")
        print("- Docker container not set up")
        print("- SSH access issues")
        print("- S3 permissions issues")

if __name__ == "__main__":
    asyncio.run(run_fast_training_test())
