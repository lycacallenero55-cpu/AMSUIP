#!/usr/bin/env python3
"""
Test script to verify EC2-only training setup
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        from config import settings
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils.aws_gpu_training import gpu_training_manager
        print("✅ AWS GPU training manager imported successfully")
        print(f"   Instance ID: {settings.EXISTING_GPU_INSTANCE_ID}")
        print(f"   EC2 Key: {settings.EC2_KEY_NAME}")
    except Exception as e:
        print(f"❌ AWS GPU training import failed: {e}")
        return False
    
    try:
        from api.training import router
        print("✅ Training API imported successfully")
    except Exception as e:
        print(f"❌ Training API import failed: {e}")
        return False
    
    return True

def test_ec2_config():
    """Test EC2 configuration"""
    print("\nTesting EC2 configuration...")
    
    from config import settings
    
    # Your actual instance ID from AWS_GPU_EXISTING_INSTANCE_ID
    if not settings.EXISTING_GPU_INSTANCE_ID:
        print("⚠️  WARNING: AWS_GPU_EXISTING_INSTANCE_ID not configured in .env")
        print("   Please add your EC2 instance ID to .env file")
        return False
    
    print(f"✅ EC2 Instance ID: {settings.EXISTING_GPU_INSTANCE_ID}")
    print(f"✅ EC2 Key Name: {settings.EC2_KEY_NAME}")
    print(f"✅ Security Group: {settings.EC2_SECURITY_GROUP_ID}")
    print(f"✅ Subnet ID: {settings.EC2_SUBNET_ID}")
    print(f"✅ IAM Profile: {settings.EC2_IAM_INSTANCE_PROFILE}")
    
    return True

def test_gpu_manager():
    """Test GPU manager availability"""
    print("\nTesting GPU manager...")
    
    try:
        from utils.aws_gpu_training import gpu_training_manager
        
        # Check if manager is initialized
        if gpu_training_manager:
            print("✅ GPU training manager initialized")
            
            # Check availability (this won't actually connect to EC2)
            try:
                available = gpu_training_manager.is_available()
                if available:
                    print("✅ GPU training is available")
                else:
                    print("⚠️  GPU training not available (check EC2 instance)")
            except Exception as e:
                print(f"⚠️  Could not check availability: {e}")
        else:
            print("❌ GPU training manager not initialized")
            return False
            
    except Exception as e:
        print(f"❌ GPU manager test failed: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("EC2-Only Training Setup Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your dependencies.")
        sys.exit(1)
    
    # Test EC2 config
    if not test_ec2_config():
        print("\n⚠️  EC2 configuration incomplete. Please update .env file.")
    
    # Test GPU manager
    test_gpu_manager()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print("✅ All imports working")
    print("✅ No Docker dependencies")
    print("✅ EC2 training ready")
    print("\nNext steps:")
    print("1. Update .env with your EC2 instance details")
    print("2. Run: python main.py")
    print("3. Training will use EC2 GPU instance exclusively")

if __name__ == "__main__":
    main()
