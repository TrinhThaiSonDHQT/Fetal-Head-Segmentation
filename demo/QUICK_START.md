# 🚀 Quick Start Guide - Fetal Head Segmentation Demo

## One-Command Launch

### Windows

```bash
cd demo
run_demo.bat
```

### Linux/Mac

```bash
cd demo
chmod +x run_demo.sh
./run_demo.sh
```

---

## What You'll See

Once launched, the application opens in your browser at **http://localhost:7860**

You'll also get a **public URL** like: `https://xxxxx.gradio.live` (shareable!)

---

## Two Main Features

### 1️⃣ **Single Image Segmentation** (Tab 1)

**Steps:**

1. Click "📸 Single Image Segmentation" tab
2. Upload an ultrasound image
3. Click "🚀 Segment Image"
4. View instant results with metrics

**Output:**

- Segmented fetal head (green overlay)
- Performance: Inference time, FPS
- Metrics: Confidence score, coverage %

**Use For:**

- Quick testing
- Live demonstrations
- Screenshot for thesis

---

### 2️⃣ **Video Stream Processing** (Tab 2)

**Steps:**

1. Click "🎬 Real-Time Video Stream" tab
2. Upload a video file (.mp4, .avi, etc.)
3. Click "🎥 Process Video"
4. Wait for processing (progress shown)
5. Download processed video

**Output:**

- Side-by-side video (original | segmented)
- Frame-by-frame metrics overlay
- Performance summary

**Use For:**

- Thesis presentation videos
- Demonstrating real-time capability
- Embedded system simulation

---

## 📸 Create Video from Images

Don't have ultrasound videos? Create one from test images:

```python
# In Python terminal
import cv2
import numpy as np
from pathlib import Path

# Load test images
image_dir = Path('shared/dataset_v4/test_set/images')
images = sorted(image_dir.glob('*.png'))[:50]  # First 50 images

# Create video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo/sample_video.mp4', fourcc, 10.0, (256, 256), False)

for img_path in images:
    frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    frame = cv2.resize(frame, (256, 256))
    out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

out.release()
print("✅ Created: demo/sample_video.mp4")
```

Then upload `sample_video.mp4` in Tab 2!

---

## 🎯 Thesis Defense Tips

### Before Defense:

1. **Test the demo** - Process 2-3 videos beforehand
2. **Take screenshots** - Both tabs for thesis report
3. **Note metrics** - Record FPS, latency for embedded analysis

### During Defense:

1. **Start with Tab 1** - Show instant single-image results
2. **Switch to Tab 2** - Play pre-processed video
3. **Let committee test** - Share public URL for hands-on

### Questions to Prepare:

- "What's the FPS?" → Check video metrics
- "Can it run on embedded devices?" → Show latency < 50ms
- "How accurate is it?" → Show confidence scores

---

## 🔧 Troubleshooting

### Port Already in Use

```python
# Edit app.py line 461
demo.launch(server_port=7861)  # Change port
```

### Model Not Found

```bash
# Verify model exists
ls best_models/best_model_mobinet_aspp_residual_se_v2.pth

# Update path in app.py if different location
```

### Slow Performance

- GPU recommended (automatic detection)
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Should print `True` for GPU acceleration

---

## 📊 Expected Performance

| Device            | Inference Time | FPS    | Real-Time? |
| ----------------- | -------------- | ------ | ---------- |
| NVIDIA RTX 3060   | 10-15ms        | 60-100 | ✅ Yes     |
| NVIDIA GTX 1650   | 20-30ms        | 30-50  | ✅ Yes     |
| CPU (i7-10th gen) | 80-120ms       | 8-12   | ⚠️ Limited |
| CPU (i5-8th gen)  | 150-200ms      | 5-7    | ❌ No      |

**Embedded Target:** <50ms for real-time ultrasound (20+ FPS)

---

## 🎬 Demo Workflow Example

```
1. Launch demo → Browser opens
2. Tab 1: Upload single image → 15ms inference → ✅
3. Tab 2: Upload 5-second video (150 frames)
   - Processing: 30 seconds
   - Output: Side-by-side video
   - Metrics: 65 FPS average
4. Download video → Add to thesis presentation
5. Share public URL → Send to committee
```

---

**Ready to impress! 🚀**
