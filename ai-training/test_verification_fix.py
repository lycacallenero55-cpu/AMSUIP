#!/usr/bin/env python3
"""
Test script to verify verification API fixes
"""

import requests
import json
from PIL import Image
import io
import numpy as np

def create_test_image():
    """Create a simple test signature image"""
    # Create a simple white image with some black lines (signature-like)
    img = Image.new('RGB', (224, 224), 'white')
    pixels = img.load()
    
    # Draw some simple lines to simulate a signature
    for i in range(50, 150):
        for j in range(30, 40):
            pixels[i, j] = (0, 0, 0)  # Black line
    
    for i in range(60, 160):
        for j in range(60, 70):
            pixels[i, j] = (0, 0, 0)  # Another black line
    
    return img

def test_verification_health():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/verification/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        print(f"Health check response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_verification_identify():
    """Test the identify endpoint"""
    try:
        # Create test image
        test_img = create_test_image()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Test identify endpoint
        files = {'test_file': ('test_signature.png', img_bytes, 'image/png')}
        response = requests.post("http://localhost:8000/api/verification/identify", files=files, timeout=30)
        
        print(f"Identify status: {response.status_code}")
        result = response.json()
        print(f"Identify response: {json.dumps(result, indent=2)}")
        
        # Check if response has expected structure
        expected_keys = ['predicted_student', 'is_match', 'confidence', 'success']
        has_expected_keys = all(key in result for key in expected_keys)
        
        return response.status_code == 200 and has_expected_keys
        
    except Exception as e:
        print(f"Identify test failed: {e}")
        return False

def test_verification_verify():
    """Test the verify endpoint"""
    try:
        # Create test image
        test_img = create_test_image()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Test verify endpoint
        files = {'test_file': ('test_signature.png', img_bytes, 'image/png')}
        data = {'student_id': 1}  # Test with student ID 1
        response = requests.post("http://localhost:8000/api/verification/verify", files=files, data=data, timeout=30)
        
        print(f"Verify status: {response.status_code}")
        result = response.json()
        print(f"Verify response: {json.dumps(result, indent=2)}")
        
        # Check if response has expected structure
        expected_keys = ['is_match', 'confidence', 'predicted_student', 'success']
        has_expected_keys = all(key in result for key in expected_keys)
        
        return response.status_code == 200 and has_expected_keys
        
    except Exception as e:
        print(f"Verify test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing verification API fixes...")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing health check...")
    health_ok = test_verification_health()
    
    # Test identify endpoint
    print("\n2. Testing identify endpoint...")
    identify_ok = test_verification_identify()
    
    # Test verify endpoint
    print("\n3. Testing verify endpoint...")
    verify_ok = test_verification_verify()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Health check: {'PASS' if health_ok else 'FAIL'}")
    print(f"Identify endpoint: {'PASS' if identify_ok else 'FAIL'}")
    print(f"Verify endpoint: {'PASS' if verify_ok else 'FAIL'}")
    
    all_passed = health_ok and identify_ok and verify_ok
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    if not all_passed:
        print("\nNote: Some tests failed. This might be expected if:")
        print("- No trained models are available")
        print("- Database is not accessible")
        print("- Server is not running on localhost:8000")
        print("\nThe important thing is that the endpoints return proper error responses instead of crashing.")

if __name__ == "__main__":
    main()