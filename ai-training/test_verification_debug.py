#!/usr/bin/env python3
"""
Debug script to test verification endpoints and identify model loading issues
"""

import requests
import json
import sys

def test_verification_endpoints():
    """Test verification endpoints to identify issues"""
    base_url = "http://localhost:8000/api/verification"
    
    print("🔍 Testing Verification System...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Database: {health_data.get('database_available', 'Unknown')}")
            print(f"   Models: {health_data.get('trained_models_count', 0)}")
            print(f"   Preprocessor: {health_data.get('preprocessor_available', False)}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Debug models
    print("\n2. Testing Model Debug...")
    try:
        response = requests.get(f"{base_url}/debug/models", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            debug_data = response.json()
            print(f"   Database Status: {debug_data.get('database_status', 'Unknown')}")
            print(f"   AI Model Info: {debug_data.get('ai_model_info', 'Unknown')}")
            print(f"   Legacy Models: {len(debug_data.get('legacy_models', []))}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Test identify endpoint with dummy image
    print("\n3. Testing Identify Endpoint...")
    try:
        # Create a simple test image
        from PIL import Image
        import io
        
        # Create a simple white image
        test_img = Image.new('RGB', (224, 224), 'white')
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {'test_file': ('test.png', img_bytes, 'image/png')}
        response = requests.post(f"{base_url}/identify", files=files, timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Success: {result.get('success', 'Unknown')}")
            print(f"   Error: {result.get('error', 'None')}")
            print(f"   Model Type: {result.get('model_type', 'Unknown')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Debug test completed!")

if __name__ == "__main__":
    test_verification_endpoints()