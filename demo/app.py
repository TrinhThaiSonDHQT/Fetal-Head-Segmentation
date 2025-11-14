"""
Unified Fetal Head Segmentation Demo Platform
Combines real-time video processing and single image inference
"""
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SegmentationDemo:
    """Unified demo handler for both video and image segmentation"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, path):
        """Load trained model"""
        try:
            from efficient_focus.src.models.mobinet_aspp_residual_se.mobinet_aspp_residual_se import MobileNetV2ASPPResidualSEUNet
            
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model = MobileNetV2ASPPResidualSEUNet(in_channels=1, out_channels=1)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            print(f"✅ Model loaded successfully from {path}")
            print(f"📊 Best Dice Score: {checkpoint.get('best_dice', 'N/A')}")
            
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess(self, frame):
        """Preprocess single frame/image to model input"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        
        # Resize to model input size (256x256)
        frame_resized = cv2.resize(frame, (256, 256))
        
        # Normalize to [0, 1]
        frame_norm = frame_resized.astype(np.float32) / 255.0
        
        # Convert to tensor [1, 1, 256, 256]
        tensor = torch.from_numpy(frame_norm).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, tensor):
        """Run inference on preprocessed tensor"""
        with torch.no_grad():
            start = time.time()
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)  # Model outputs logits
            mask = (probs > 0.5).float()
            inference_time = (time.time() - start) * 1000  # ms
        
        return mask.cpu().numpy()[0, 0], probs.cpu().numpy()[0, 0], inference_time
    
    def create_overlay(self, original, mask, add_metrics=True, inference_time=None):
        """Create segmentation overlay with contours"""
        # Ensure original is in correct format
        if len(original.shape) == 2:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            original_bgr = original.copy()
        
        # Resize mask to original size
        mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        # Create colored overlay (green for fetal head)
        overlay = original_bgr.copy()
        overlay[mask_resized > 0.5] = [0, 255, 0]
        
        # Blend original with overlay
        result = cv2.addWeighted(original_bgr, 0.7, overlay, 0.3, 0)
        
        # Add contours
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # Add performance metrics overlay
        if add_metrics and inference_time is not None:
            fps = 1000 / inference_time if inference_time > 0 else 0
            cv2.putText(result, f"Latency: {inference_time:.1f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result
    
    def segment_single_image(self, image):
        """
        Process single image upload
        Returns: (result_image, metrics_text)
        """
        if image is None:
            return None, "⚠️ Please upload an image"
        
        try:
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Preprocess and predict
            tensor = self.preprocess(img_array)
            mask, probs, inference_time = self.predict(tensor)
            
            # Create visualization
            result = self.create_overlay(img_array, mask, add_metrics=False)
            
            # Calculate metrics
            dice_score = probs.mean()  # Approximation
            segmented_area = (mask > 0.5).sum()
            total_pixels = mask.size
            coverage = (segmented_area / total_pixels) * 100
            
            # Format metrics
            metrics = f"""
⚡ **Performance Metrics:**
- Inference Time: **{inference_time:.2f} ms**
- FPS: **{1000/inference_time:.1f}**
- Device: **{self.device}**

🎯 **Segmentation Metrics:**
- Avg Confidence: **{dice_score:.4f}**
- Coverage: **{coverage:.2f}%**
- Segmented Pixels: **{int(segmented_area):,}**

✅ **Status:** Successfully segmented fetal head region
            """
            
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), metrics
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def process_video_stream(self, video_file, progress=gr.Progress()):
        """
        Process video file frame-by-frame
        Returns: output_video_path, summary_text
        """
        if video_file is None:
            return None, "⚠️ Please upload a video file"
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_file)
            
            if not cap.isOpened():
                return None, "❌ Failed to open video file"
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output video writer
            output_path = 'demo/output_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Side-by-side output (original + overlay)
            out_width = width * 2
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))
            
            # Processing
            frame_idx = 0
            inference_times = []
            
            progress(0, desc="Starting video processing...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for processing
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Inference
                tensor = self.preprocess(frame_gray)
                mask, probs, inf_time = self.predict(tensor)
                inference_times.append(inf_time)
                
                # Create overlay
                overlay = self.create_overlay(frame, mask, add_metrics=True, inference_time=inf_time)
                
                # Side-by-side visualization
                combined = np.hstack([frame, overlay])
                
                # Add frame counter
                cv2.putText(combined, f"Frame: {frame_idx+1}/{total_frames}", (10, height-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                out.write(combined)
                frame_idx += 1
                
                # Update progress
                progress((frame_idx / total_frames), desc=f"Processing frame {frame_idx}/{total_frames}")
            
            # Cleanup
            cap.release()
            out.release()
            
            # Generate summary
            avg_inference = np.mean(inference_times)
            avg_fps = 1000 / avg_inference
            
            summary = f"""
🎬 **Video Processing Complete!**

📊 **Performance Summary:**
- Total Frames: **{frame_idx}**
- Video Duration: **{frame_idx/fps:.2f}s**
- Average Inference Time: **{avg_inference:.2f} ms**
- Average FPS: **{avg_fps:.1f}**
- Min/Max Latency: **{np.min(inference_times):.1f} / {np.max(inference_times):.1f} ms**

🔧 **Processing Details:**
- Original Resolution: **{width}x{height}**
- Model Input Size: **256x256**
- Output Format: **Side-by-side (original + segmentation)**
- Device: **{self.device}**

✅ **Output saved to:** `{output_path}`
            """
            
            return output_path, summary
            
        except Exception as e:
            return None, f"❌ Error processing video: {str(e)}"


# Initialize model
MODEL_PATH = 'best_models/best_model_mobinet_aspp_residual_se_v2.pth'
demo_app = SegmentationDemo(MODEL_PATH)


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
#subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 20px;
}
.metric-box {
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}
"""

# Create Gradio Blocks interface
with gr.Blocks(css=custom_css, title="Fetal Head Segmentation Demo") as demo:
    
    gr.Markdown(
        """
        <div id="title">
        <h1>🤰 Fetal Head Segmentation System</h1>
        <p>MobileNetV2-Based ASPP Residual SE U-Net</p>
        </div>
        """,
        elem_id="title"
    )
    
    gr.Markdown(
        """
        <div id="subtitle">
        <b>Interactive Demo Platform</b> - Upload single images or process entire ultrasound video sequences in real-time
        </div>
        """,
        elem_id="subtitle"
    )
    
    # Create tabs for two features
    with gr.Tabs():
        
        # ====================================================================
        # TAB 1: Single Image Segmentation
        # ====================================================================
        with gr.Tab("📸 Single Image Segmentation"):
            gr.Markdown(
                """
                ### Upload Ultrasound Image
                Upload a single ultrasound image to see instant segmentation results with performance metrics.
                Supports: `.png`, `.jpg`, `.jpeg`, `.bmp`
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Ultrasound Image",
                        height=400
                    )
                    image_submit_btn = gr.Button("🚀 Segment Image", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    image_output = gr.Image(
                        type="pil",
                        label="Segmentation Result",
                        height=400
                    )
            
            with gr.Row():
                image_metrics = gr.Markdown(
                    "⏳ Upload an image and click 'Segment Image' to see results...",
                    elem_classes=["metric-box"]
                )
            
            # Example images
            gr.Markdown("### 📚 Example Images")
            gr.Examples(
                examples=[
                    ["shared/dataset_v4/test_set/images/000.png"],
                    ["shared/dataset_v4/test_set/images/001.png"],
                    ["shared/dataset_v4/test_set/images/002.png"],
                    ["shared/dataset_v4/test_set/images/003.png"],
                ],
                inputs=image_input,
                label="Click to load example"
            )
            
            # Connect button
            image_submit_btn.click(
                fn=demo_app.segment_single_image,
                inputs=image_input,
                outputs=[image_output, image_metrics]
            )
        
        # ====================================================================
        # TAB 2: Video Stream Processing
        # ====================================================================
        with gr.Tab("🎬 Real-Time Video Stream"):
            gr.Markdown(
                """
                ### Process Ultrasound Video Sequence
                Upload a video file to process frame-by-frame with real-time performance metrics.
                Supports: `.mp4`, `.avi`, `.mov`, `.mkv`
                
                **Note:** Processing may take time depending on video length. Progress will be shown below.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="Upload Video File",
                        height=400
                    )
                    video_submit_btn = gr.Button("🎥 Process Video", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="Processed Video (Side-by-side)",
                        height=400
                    )
            
            with gr.Row():
                video_metrics = gr.Markdown(
                    "⏳ Upload a video and click 'Process Video' to start...",
                    elem_classes=["metric-box"]
                )
            
            # Connect button
            video_submit_btn.click(
                fn=demo_app.process_video_stream,
                inputs=video_input,
                outputs=[video_output, video_metrics]
            )
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    gr.Markdown(
        """
        ---
        ### 📄 About This Demo
        
        **Model Architecture:** MobileNetV2-ASPP-ResidualSE U-Net
        - **Encoder:** Pre-trained MobileNetV2 (ImageNet transfer learning)
        - **Bottleneck:** ASPP (Atrous Spatial Pyramid Pooling) for multi-scale features
        - **Decoder:** Residual blocks with Squeeze-and-Excitation attention
        - **Training Dataset:** HC18 Grand Challenge (999 training images)
        - **Target Metrics:** DSC ≥97.81%, mIoU ≥97.90%
        
        **Performance:**
        - Inference Time: ~10-30ms per frame (GPU)
        - Model Size: ~15MB
        - Input Size: 256×256 grayscale
        
        ---
        *Developed for Final Year IT Thesis Project - Fetal Head Segmentation in Ultrasound Images*
        """
    )


# ============================================================================
# LAUNCH APP
# ============================================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Create public URL
        show_error=True,
        show_api=False
    )
