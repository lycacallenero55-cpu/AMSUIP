#!/usr/bin/env python3
"""
AWS GPU Training Setup Verification Script
Run this to check if your AWS setup is working correctly
"""

import os
import boto3
import sys
from botocore.exceptions import ClientError, NoCredentialsError

def test_aws_credentials():
    """Test if AWS credentials are configured"""
    print("🔍 Testing AWS Credentials...")
    try:
        # Test with a simple S3 call
        s3_client = boto3.client('s3')
        s3_client.list_buckets()
        print("   ✅ AWS credentials are working")
        return True
    except NoCredentialsError:
        print("   ❌ AWS credentials not found")
        print("   💡 Run 'aws configure' to set up your credentials")
        return False
    except Exception as e:
        print(f"   ❌ AWS credentials error: {e}")
        return False

def test_s3_bucket():
    """Test if S3 bucket exists and is accessible"""
    print("\n🔍 Testing S3 Bucket...")
    bucket_name = os.getenv('S3_BUCKET')
    if not bucket_name:
        print("   ❌ S3_BUCKET not set in .env file")
        return False
    
    try:
        s3_client = boto3.client('s3')
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"   ✅ S3 bucket '{bucket_name}' is accessible")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"   ❌ S3 bucket '{bucket_name}' does not exist")
            print(f"   💡 Create it with: aws s3 mb s3://{bucket_name}")
        else:
            print(f"   ❌ S3 bucket error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ S3 bucket error: {e}")
        return False

def test_ec2_permissions():
    """Test if EC2 permissions are working"""
    print("\n🔍 Testing EC2 Permissions...")
    try:
        ec2_client = boto3.client('ec2')
        # Test with a simple call
        ec2_client.describe_regions(MaxResults=1)
        print("   ✅ EC2 permissions are working")
        return True
    except Exception as e:
        print(f"   ❌ EC2 permissions error: {e}")
        print("   💡 Make sure your AWS user has EC2 permissions")
        return False

def test_iam_role():
    """Test if IAM role exists"""
    print("\n🔍 Testing IAM Role...")
    try:
        iam_client = boto3.client('iam')
        role_name = 'EC2-S3-Access'
        iam_client.get_role(RoleName=role_name)
        print(f"   ✅ IAM role '{role_name}' exists")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            print(f"   ❌ IAM role 'EC2-S3-Access' does not exist")
            print("   💡 Create it with the commands in the setup guide")
        else:
            print(f"   ❌ IAM role error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ IAM role error: {e}")
        return False

def test_security_group():
    """Test if security group exists"""
    print("\n🔍 Testing Security Group...")
    try:
        ec2_client = boto3.client('ec2')
        response = ec2_client.describe_security_groups(
            GroupNames=['gpu-training-sg']
        )
        if response['SecurityGroups']:
            sg_id = response['SecurityGroups'][0]['GroupId']
            print(f"   ✅ Security group 'gpu-training-sg' exists (ID: {sg_id})")
            return True
        else:
            print("   ❌ Security group 'gpu-training-sg' not found")
            return False
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidGroup.NotFound':
            print("   ❌ Security group 'gpu-training-sg' does not exist")
            print("   💡 Create it with the commands in the setup guide")
        else:
            print(f"   ❌ Security group error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Security group error: {e}")
        return False

def test_key_pair():
    """Test if key pair exists"""
    print("\n🔍 Testing Key Pair...")
    try:
        ec2_client = boto3.client('ec2')
        response = ec2_client.describe_key_pairs(
            KeyNames=['gpu-training-key']
        )
        if response['KeyPairs']:
            print("   ✅ Key pair 'gpu-training-key' exists")
            return True
        else:
            print("   ❌ Key pair 'gpu-training-key' not found")
            return False
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidKeyPair.NotFound':
            print("   ❌ Key pair 'gpu-training-key' does not exist")
            print("   💡 Create it with the commands in the setup guide")
        else:
            print(f"   ❌ Key pair error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Key pair error: {e}")
        return False

def test_gpu_availability():
    """Test if GPU instances are available in your region"""
    print("\n🔍 Testing GPU Instance Availability...")
    try:
        ec2_client = boto3.client('ec2')
        response = ec2_client.describe_instance_types(
            InstanceTypes=['g4dn.xlarge']
        )
        if response['InstanceTypes']:
            print("   ✅ GPU instances (g4dn.xlarge) are available in your region")
            return True
        else:
            print("   ❌ GPU instances not available in your region")
            return False
    except Exception as e:
        print(f"   ❌ GPU availability check failed: {e}")
        return False

def check_env_variables():
    """Check if all required environment variables are set"""
    print("\n🔍 Checking Environment Variables...")
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY', 
        'AWS_REGION',
        'S3_BUCKET',
        'AWS_KEY_NAME',
        'AWS_SECURITY_GROUP_ID',
        'AWS_SUBNET_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"   ❌ Missing environment variables: {', '.join(missing_vars)}")
        print("   💡 Add them to your .env file")
        return False
    else:
        print("   ✅ All required environment variables are set")
        return True

def main():
    """Run all tests"""
    print("🚀 AWS GPU Training Setup Verification")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    tests = [
        check_env_variables,
        test_aws_credentials,
        test_s3_bucket,
        test_ec2_permissions,
        test_iam_role,
        test_security_group,
        test_key_pair,
        test_gpu_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AWS setup is ready for GPU training!")
        print("\n✅ You can now:")
        print("   • Start GPU training from your frontend")
        print("   • See real-time training logs")
        print("   • Enjoy 10-50x faster training")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\n💡 Next steps:")
        print("   1. Follow the AWS_GPU_SETUP_SIMPLE.md guide")
        print("   2. Run this script again to verify")
        print("   3. Contact support if issues persist")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)