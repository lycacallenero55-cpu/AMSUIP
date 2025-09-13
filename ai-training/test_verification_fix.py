#!/usr/bin/env python3
"""
Test script to verify the verification fix
"""
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def create_test_image():
    """Create a simple test signature image"""
    # Create a simple black signature on white background
    img = Image.new('RGB', (224, 224), 'white')
    pixels = np.array(img)
    
    # Draw a simple signature-like pattern
    for i in range(50, 150):
        for j in range(50, 100):
            pixels[i, j] = [0, 0, 0]  # Black
    
    for i in range(100, 200):
        for j in range(100, 150):
            pixels[i, j] = [0, 0, 0]  # Black
    
    img = Image.fromarray(pixels)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_verification():
    """Test the verification endpoint"""
    print("🧪 Testing Verification Fix...")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get("http://localhost:8000/api/verification/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test 2: Create test images
    print("\n2. Creating test images...")
    test_images = []
    for i in range(3):
        img_str = create_test_image()
        test_images.append(img_str)
        print(f"✅ Created test image {i+1}")
    
    # Test 3: Test verification with different images
    print("\n3. Testing verification with different images...")
    results = []
    
    for i, img_str in enumerate(test_images):
        print(f"\nTesting image {i+1}...")
        try:
            payload = {
                "image": img_str,
                "student_id": None  # Let it identify
            }
            
            response = requests.post(
                "http://localhost:8000/api/verification/identify",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append(result)
                
                print(f"✅ Image {i+1} result:")
                print(f"   - Predicted student: {result.get('predicted_student', {}).get('name', 'Unknown')}")
                print(f"   - Student ID: {result.get('predicted_student_id', 0)}")
                print(f"   - Is match: {result.get('is_match', False)}")
                print(f"   - Is unknown: {result.get('is_unknown', True)}")
                print(f"   - Confidence: {result.get('confidence', 0.0):.3f}")
                print(f"   - Student confidence: {result.get('student_confidence', 0.0):.3f}")
                print(f"   - Global score: {result.get('global_score', 0.0):.3f}")
                print(f"   - Global margin: {result.get('global_margin', 0.0):.3f}")
            else:
                print(f"❌ Image {i+1} failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Image {i+1} error: {e}")
    
    # Test 4: Check for diversity in results
    print("\n4. Analyzing results...")
    if len(results) >= 2:
        student_ids = [r.get('predicted_student_id', 0) for r in results]
        student_names = [r.get('predicted_student', {}).get('name', 'Unknown') for r in results]
        
        print(f"Student IDs: {student_ids}")
        print(f"Student names: {student_names}")
        
        # Check if we're getting different predictions
        unique_ids = set(student_ids)
        unique_names = set(student_names)
        
        if len(unique_ids) > 1:
            print("✅ Different student IDs detected - fix is working!")
        elif len(unique_names) > 1:
            print("✅ Different student names detected - fix is working!")
        else:
            print("⚠️  Still getting same predictions - may need further tuning")
            
        # Check if we're getting valid predictions
        valid_predictions = [r for r in results if r.get('predicted_student_id', 0) > 0]
        print(f"Valid predictions: {len(valid_predictions)}/{len(results)}")
        
        if len(valid_predictions) > 0:
            print("✅ Getting valid predictions!")
        else:
            print("⚠️  No valid predictions - may need to retrain models")
    else:
        print("❌ Not enough results to analyze")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_verification()