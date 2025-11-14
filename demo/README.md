# Fetal Head Segmentation Demo Platform

**Interactive web-based demonstration system** combining single image segmentation and real-time video stream processing.

## 🌟 Features

### 1. **Single Image Segmentation** 📸

- Upload individual ultrasound images
- Instant segmentation with overlay visualization
- Performance metrics (inference time, FPS, confidence)
- Side-by-side comparison

### 2. **Real-Time Video Stream Processing** 🎬

- Upload entire ultrasound video sequences
- Frame-by-frame processing with live metrics
- Side-by-side output (original + segmentation)
- Progress tracking and performance summary

## 🚀 Quick Start

### Installation

```bash
# Navigate to demo directory
cd demo

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Launch web interface
python app.py
```

The application will start on `http://localhost:7860` and generate a **public shareable URL** (via Gradio).

**Output:**

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
```

Share the public URL with reviewers/committee members!

## 📋 Usage Guide

### Single Image Mode

1. Click **"📸 Single Image Segmentation"** tab
2. Upload an ultrasound image (`.png`, `.jpg`, `.jpeg`)
3. Click **"🚀 Segment Image"**
4. View results:
   - Segmented fetal head (green overlay + contours)
   - Performance metrics (inference time, FPS)
   - Segmentation confidence and coverage

### Video Stream Mode

1. Click **"🎬 Real-Time Video Stream"** tab
2. Upload a video file (`.mp4`, `.avi`, `.mov`)
3. Click **"🎥 Process Video"**
4. Wait for processing (progress bar shown)
5. Download processed video:
   - Left: Original frames
   - Right: Segmentation overlay with metrics
   - Bottom: Frame counter

## 🔧 Technical Details

### Model Architecture

- **Name:** MobileNetV2-ASPP-ResidualSE U-Net
- **Encoder:** Pre-trained MobileNetV2 (ImageNet transfer learning)
- **Bottleneck:** ASPP (Atrous Spatial Pyramid Pooling)
- **Decoder:** Residual blocks with SE attention
- **Input Size:** 256×256 grayscale
- **Output:** Binary segmentation mask

### Performance

- **Inference Time:** ~10-30ms per frame (GPU) / ~50-100ms (CPU)
- **FPS:** 30-100 (GPU) / 10-20 (CPU)
- **Model Size:** ~15MB
- **Device Support:** CUDA GPU (automatic fallback to CPU)

### Supported Formats

- **Images:** PNG, JPG, JPEG, BMP
- **Videos:** MP4, AVI, MOV, MKV

## 📁 File Structure

```
demo/
├── app.py                  # Main application
├── requirements.txt        # Dependencies
├── README.md              # This file
└── output_video.mp4       # Generated after video processing
```

## 🎯 Use Cases

### For Thesis Defense

1. **Live Demonstration:** Show committee members the interactive demo
2. **Video Presentation:** Process sample videos beforehand for slides
3. **Q&A Support:** Answer questions by testing on custom images

### For Documentation

1. **Screenshots:** Capture results for thesis report
2. **Performance Metrics:** Export timing/accuracy data
3. **Public Sharing:** Include Gradio URL in appendix

### For Embedded System Simulation

- **FPS Metrics:** Demonstrate real-time capability
- **Latency Analysis:** Show inference time consistency
- **Scalability:** Process various video lengths/resolutions

## ⚙️ Configuration

### Change Model

Edit `app.py` line 209:

```python
MODEL_PATH = 'path/to/your/model.pth'
```

### Change Port

Edit `app.py` line 461:

```python
demo.launch(server_port=7860)  # Change to desired port
```

### Disable Public URL

Edit `app.py` line 461:

```python
demo.launch(share=False)  # No public URL generation
```

## 🐛 Troubleshooting

### "Model not found" error

- Verify model path in `app.py` (line 209)
- Ensure model file exists: `best_models/best_model_mobinet_aspp_residual_se_v2.pth`

### Slow video processing

- Use GPU if available (automatic detection)
- Reduce video resolution/length for faster testing
- Check CUDA availability: `torch.cuda.is_available()`

### CUDA out of memory

- Reduce batch size (currently 1 frame at a time)
- Use CPU mode by setting `device='cpu'` in `app.py` line 208

## 📊 Example Outputs

### Single Image

```
⚡ Performance Metrics:
- Inference Time: 12.34 ms
- FPS: 81.0
- Device: cuda

🎯 Segmentation Metrics:
- Avg Confidence: 0.9782
- Coverage: 23.45%
- Segmented Pixels: 15,360
```

### Video Stream

```
🎬 Video Processing Complete!

📊 Performance Summary:
- Total Frames: 150
- Video Duration: 5.00s
- Average Inference Time: 15.23 ms
- Average FPS: 65.7
- Min/Max Latency: 12.1 / 18.9 ms
```

## 📝 License

Part of the **Fetal Head Segmentation in Ultrasound Images** thesis project.

---

**Questions?** Check the main project README or contact the author.
