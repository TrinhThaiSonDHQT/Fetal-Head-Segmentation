"""
Test Script for Backend Error Handling

This script tests various error scenarios to verify that the backend
handles them gracefully and returns appropriate error messages.
"""
import requests
import io
from PIL import Image
import numpy as np

BASE_URL = "http://localhost:5000/api"


def test_health_check():
    """Test health check endpoint."""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ Health check failed: {e}")


def test_no_file_upload():
    """Test uploading without a file."""
    print("\n=== Testing Upload Without File ===")
    try:
        response = requests.post(f"{BASE_URL}/upload", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 400
        print("✓ Correctly rejected empty upload")
    except Exception as e:
        print(f"✗ Test failed: {e}")


def test_corrupted_image():
    """Test uploading corrupted image data."""
    print("\n=== Testing Corrupted Image ===")
    try:
        # Create corrupted image data
        corrupted_data = b"This is not a valid image file"
        files = {'image': ('corrupted.jpg', io.BytesIO(corrupted_data), 'image/jpeg')}
        
        response = requests.post(f"{BASE_URL}/upload", files=files, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 400
        print("✓ Correctly rejected corrupted image")
    except Exception as e:
        print(f"✗ Test failed: {e}")


def test_valid_image():
    """Test uploading a valid image."""
    print("\n=== Testing Valid Image Upload ===")
    try:
        # Create a simple test image
        img = Image.new('RGB', (256, 256), color='gray')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {'image': ('test.png', img_bytes, 'image/png')}
        
        response = requests.post(f"{BASE_URL}/upload", files=files, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Inference time: {data.get('inference_time')}ms")
            print(f"Confidence: {data.get('confidence_score')}")
            print("✓ Valid image processed successfully")
        else:
            print(f"Response: {response.json()}")
            print("✗ Unexpected response")
            
    except requests.Timeout:
        print("✗ Request timeout (expected for slow inference)")
    except Exception as e:
        print(f"✗ Test failed: {e}")


def test_large_file():
    """Test uploading a file larger than the limit."""
    print("\n=== Testing Large File Upload ===")
    try:
        # Create a large image (>16MB)
        # 5000x5000 RGB = ~75MB uncompressed
        img = Image.new('RGB', (5000, 5000), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG', compress_level=0)
        img_bytes.seek(0)
        
        print(f"Image size: {len(img_bytes.getvalue()) / 1024 / 1024:.2f} MB")
        
        files = {'image': ('large.png', img_bytes, 'image/png')}
        
        response = requests.post(f"{BASE_URL}/upload", files=files, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 413:
            print("✓ Correctly rejected large file")
        else:
            print("✗ Should have rejected large file")
            
    except Exception as e:
        print(f"Test completed with exception: {e}")


def test_timeout():
    """Test request timeout handling."""
    print("\n=== Testing Request Timeout ===")
    try:
        # Very short timeout to trigger timeout error
        img = Image.new('RGB', (256, 256), color='gray')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {'image': ('test.png', img_bytes, 'image/png')}
        
        response = requests.post(f"{BASE_URL}/upload", files=files, timeout=0.001)
        print(f"Response: {response.json()}")
        
    except requests.Timeout:
        print("✓ Timeout triggered as expected (client-side)")
    except Exception as e:
        print(f"Test result: {e}")


def test_invalid_endpoint():
    """Test accessing invalid endpoint."""
    print("\n=== Testing Invalid Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/invalid", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 404
        print("✓ Correctly returned 404 for invalid endpoint")
    except Exception as e:
        print(f"✗ Test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Backend Error Handling Test Suite")
    print("=" * 60)
    print("\nMake sure the Flask server is running on http://localhost:5000")
    print("Press Enter to continue...")
    input()
    
    # Run all tests
    test_health_check()
    test_no_file_upload()
    test_corrupted_image()
    test_invalid_endpoint()
    test_valid_image()
    # test_large_file()  # Commented out as it takes time to generate
    # test_timeout()     # Commented out as it's client-side
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)
