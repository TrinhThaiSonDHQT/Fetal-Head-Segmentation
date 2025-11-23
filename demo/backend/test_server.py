"""
Quick test script to verify Flask server is working properly.
Run this to test the backend without the frontend.
"""
import requests
import json
from pathlib import Path

# Server URL
BASE_URL = "http://localhost:5000/api"

def test_health():
    """Test health check endpoint."""
    print("Testing /api/health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_upload():
    """Test image upload endpoint."""
    print("\nTesting /api/upload...")
    
    # Find a test image
    test_images = list(Path("../../shared/dataset_v4/test_set/images").glob("*.png"))
    if not test_images:
        print("✗ No test images found")
        return False
    
    test_image = test_images[0]
    print(f"Using test image: {test_image.name}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'image': f}
            data = {'use_tta': 'true'}
            response = requests.post(
                f"{BASE_URL}/upload",
                files=files,
                data=data,
                timeout=30
            )
        
        print(f"✓ Status: {response.status_code}")
        result = response.json()
        
        if result.get('success'):
            print("✓ Upload successful")
            print(f"  - Inference time: {result.get('inference_time')}ms")
            print(f"  - Confidence: {result.get('confidence_score', 'N/A')}")
            print(f"  - Valid ultrasound: {result.get('is_valid_ultrasound', 'N/A')}")
            if result.get('warnings'):
                print(f"  - Warnings: {result.get('warnings')}")
        else:
            print(f"✗ Upload failed: {result.get('error')}")
            
        return result.get('success', False)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Backend Server Test")
    print("="*60)
    print("Make sure the Flask server is running (python app.py)")
    print("="*60 + "\n")
    
    # Run tests
    health_ok = test_health()
    
    if health_ok:
        upload_ok = test_upload()
        
        print("\n" + "="*60)
        if health_ok and upload_ok:
            print("✓ All tests passed")
        else:
            print("✗ Some tests failed")
        print("="*60)
    else:
        print("\n✗ Health check failed - is the server running?")
