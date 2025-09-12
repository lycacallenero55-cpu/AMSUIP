#!/usr/bin/env python3
"""
Simple test to check if the server is running and responding
"""
import urllib.request
import urllib.parse
import json

def test_server():
    """Test if the server is running"""
    print("🧪 Testing Server...")
    print("=" * 30)
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        with urllib.request.urlopen("http://localhost:8000/api/verification/health", timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"✅ Health check passed: {data}")
            else:
                print(f"❌ Health check failed: {response.status}")
                return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    print("\n2. Testing identify endpoint...")
    try:
        # Create a simple test payload
        payload = {
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",  # 1x1 pixel
            "student_id": None
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:8000/api/verification/identify",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                result = json.loads(response.read().decode())
                print(f"✅ Identify endpoint responded:")
                print(f"   - Predicted student: {result.get('predicted_student', {}).get('name', 'Unknown')}")
                print(f"   - Student ID: {result.get('predicted_student_id', 0)}")
                print(f"   - Is match: {result.get('is_match', False)}")
                print(f"   - Is unknown: {result.get('is_unknown', True)}")
                print(f"   - Confidence: {result.get('confidence', 0.0):.3f}")
                print(f"   - Student confidence: {result.get('student_confidence', 0.0):.3f}")
                print(f"   - Global score: {result.get('global_score', 0.0):.3f}")
                print(f"   - Global margin: {result.get('global_margin', 0.0):.3f}")
                
                # Check if global margin is zero (the problem we're fixing)
                global_margin = result.get('global_margin', 0.0)
                if global_margin <= 1e-6:
                    print("⚠️  Global margin is zero - this should be rejected now")
                else:
                    print("✅ Global margin is non-zero - good!")
                    
            else:
                print(f"❌ Identify endpoint failed: {response.status}")
                print(f"Response: {response.read().decode()}")
                
    except Exception as e:
        print(f"❌ Identify endpoint error: {e}")
    
    print("\n" + "=" * 30)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_server()