"""
Flask REST API for Fetal Head Segmentation

Provides endpoints for:
- Health check
- Image upload and segmentation
- Video stream demo with Server-Sent Events
"""
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import time
import base64
from pathlib import Path
import glob
import json

from model_loader import ModelLoader
from inference import InferenceEngine
from utils import pil_to_numpy, create_overlay, image_to_base64, numpy_to_pil
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
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
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader is not None,
        'device': str(model_loader.device) if model_loader else None
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Upload and segment ultrasound image.
    
    Expects:
        - Form data with 'image' file field
    
    Returns:
        JSON with:
        - success: bool
        - original: Base64 encoded original image
        - segmentation: Base64 encoded overlay visualization
        - inference_time: Processing time in milliseconds
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
        
        # Read image
        image = Image.open(file.stream)
        
        # Convert to RGB if needed (some images might be RGBA or other formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy for processing
        image_np = pil_to_numpy(image)
        
        # Run inference
        start_time = time.time()
        result = inference_engine.predict(image_np)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Extract results
        mask = result['mask']  # Binary mask (H, W)
        
        # Create visualization overlay
        visualization = create_overlay(image_np, mask)
        
        # Convert to base64 for JSON response
        original_b64 = image_to_base64(image_np)
        segmentation_b64 = image_to_base64(visualization)
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'segmentation': segmentation_b64,
            'inference_time': round(inference_time, 2)
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stream', methods=['GET'])
def stream_demo():
    """
    Server-Sent Events endpoint for video stream demo.
    
    Streams pre-recorded ultrasound frames with segmentation.
    Each frame is processed and sent as an SSE event.
    
    Returns:
        SSE stream with events:
        - type: 'frame' - Contains original and segmentation
        - type: 'complete' - Stream finished
        - type: 'error' - Error occurred
    """
    def generate():
        try:
            # Get all demo frames
            if not DEMO_FRAMES_DIR.exists():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Demo frames directory not found'})}\n\n"
                return
            
            # Find all image files (support multiple formats)
            frame_files = sorted(glob.glob(str(DEMO_FRAMES_DIR / 'frame_*.png'))) + \
                         sorted(glob.glob(str(DEMO_FRAMES_DIR / 'frame_*.jpg')))
            
            if not frame_files:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No demo frames found'})}\n\n"
                return
            
            # Process and stream each frame
            for idx, frame_path in enumerate(frame_files):
                # Load frame
                image = Image.open(frame_path)
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_np = pil_to_numpy(image)
                
                # Run inference
                result = inference_engine.predict(image_np)
                mask = result['mask']
                
                # Create visualization overlay
                visualization = create_overlay(image_np, mask)
                
                # Convert to base64
                original_b64 = image_to_base64(image_np)
                segmentation_b64 = image_to_base64(visualization)
                
                # Send frame event
                event_data = {
                    'type': 'frame',
                    'original': original_b64,
                    'segmentation': segmentation_b64,
                    'frame_number': idx + 1,
                    'total_frames': len(frame_files)
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Small delay to simulate real-time scanning (~10 FPS)
                time.sleep(0.1)
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
        except Exception as e:
            print(f"Error in stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


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
    print(f"Stream: GET http://localhost:5000/api/stream")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
