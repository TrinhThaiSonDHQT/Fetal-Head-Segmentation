"""
Flask REST API for Fetal Head Segmentation

Provides endpoints for:
- Health check
- Image upload and segmentation
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from pathlib import Path

from model_loader import ModelLoader
from inference import InferenceEngine
from utils import pil_to_numpy, create_overlay, image_to_base64, numpy_to_pil
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
# CORS(app)  # Enable CORS for React frontend - TEMPORARILY DISABLED FOR DEBUGGING

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.config['REQUEST_TIMEOUT'] = 30  # 30 seconds timeout for requests
MODEL_PATH = Path(__file__).parent / 'best_model_mobinet_aspp_residual_se_v2.pth'
DEMO_FRAMES_DIR = Path(__file__).parent.parent / 'frontend' / 'public' / 'demo_videos'

# Global model loader (singleton pattern)
model_loader = None
inference_engine = None


def initialize_model():
    """Initialize model loader and inference engine."""
    global model_loader, inference_engine
    
    if model_loader is None:
        print("Initializing model...")
        model_loader = ModelLoader(MODEL_PATH)
        inference_engine = InferenceEngine(model_loader.model, model_loader.device)
        print("✓ Model ready for inference")


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with status and model loading state
    """
    print("Health check endpoint called")  # Debug
    try:
        print("Building response...")  # Debug
        model_status = model_loader is not None
        device_info = str(model_loader.device) if model_loader else None
        
        response_data = {
            'status': 'healthy',
            'model_loaded': model_status,
            'device': device_info
        }
        print(f"Response data: {response_data}")  # Debug
        
        result = jsonify(response_data)
        print("Jsonify successful, returning...")  # Debug
        return result
    except Exception as e:
        print(f"ERROR in health_check: {e}")  # Debug
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Upload and segment ultrasound image.
    
    Expects:
        - Form data with 'image' file field
        - Optional 'use_tta' field (boolean, default: true)
    
    Returns:
        JSON with:
        - success: bool
        - original: Base64 encoded original image
        - segmentation: Base64 encoded overlay visualization
        - inference_time: Processing time in milliseconds
        - tta_variance: (if TTA enabled) Prediction variance
        - tta_confidence: (if TTA enabled) TTA-based confidence
        - error: Error message (if failed)
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Get TTA flag (default: True)
        use_tta = request.form.get('use_tta', 'true').lower() == 'true'
        
        # Read and validate image
        try:
            image = Image.open(file.stream)
            # Verify image is not corrupted by loading it
            image.verify()
            # Re-open after verify (verify() closes the file)
            file.stream.seek(0)
            image = Image.open(file.stream)
        except (IOError, OSError) as e:
            return jsonify({
                'success': False,
                'error': 'Corrupted or invalid image file. Please upload a valid image.'
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read image: {str(e)}'
            }), 400
        
        # Convert to RGB if needed (some images might be RGBA or other formats)
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to convert image format: {str(e)}'
            }), 400
        
        # Convert to numpy for processing
        try:
            image_np = pil_to_numpy(image)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to process image data: {str(e)}'
            }), 400
        
        # Run inference with validation (TTA enabled by default)
        try:
            result = inference_engine.process_image(image_np, use_tta=use_tta)
        except RuntimeError as e:
            return jsonify({
                'success': False,
                'error': 'Model inference failed. This may be due to GPU memory issues or invalid image dimensions.'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Inference error: {str(e)}'
            }), 500
        
        # Extract results
        mask = result['mask']  # Binary mask (H, W)
        inference_time = result['inference_time']  # Time in ms
        
        # Create visualization overlay
        try:
            visualization = create_overlay(image_np, mask)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to create visualization: {str(e)}'
            }), 500
        
        # Convert to base64 for JSON response
        original_b64 = image_to_base64(image_np)
        segmentation_b64 = image_to_base64(visualization)
        
        response_data = {
            'success': True,
            'original': original_b64,
            'segmentation': segmentation_b64,
            'inference_time': round(inference_time, 2),
            
            # Add validation data
            'is_valid_ultrasound': result['is_valid_ultrasound'],
            'confidence_score': float(result['confidence_score']),
            'quality_metrics': result['quality_metrics'],
            'warnings': result['warnings']
        }
        
        # Add TTA-specific metrics if used
        if use_tta and 'tta_variance' in result:
            response_data['tta_variance'] = result['tta_variance']
            response_data['tta_confidence'] = result['tta_confidence']
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'warnings': ['An error occurred during processing']
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16 MB.'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(408)
def request_timeout(error):
    """Handle request timeout errors."""
    return jsonify({
        'success': False,
        'error': 'Request timeout. The operation took too long to complete.'
    }), 408


if __name__ == '__main__':
    # Initialize model before starting server
    initialize_model()
    
    # Run Flask development server
    print("\n" + "="*60)
    print("Starting Fetal Head Segmentation API Server")
    print("="*60)
    print(f"Server: http://localhost:5000")
    print(f"Health: http://localhost:5000/api/health")
    print(f"Upload: POST http://localhost:5000/api/upload")
    print("="*60 + "\n")
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
    except Exception as e:
        print(f"Server crashed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Server shutting down...")
