"""
Create sample video from test images for demo purposes
Useful when you don't have actual ultrasound videos
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def create_video_from_images(
    image_dir,
    output_path,
    fps=10,
    max_frames=100,
    target_size=(256, 256)
):
    """
    Create video from image sequence
    
    Args:
        image_dir: Path to directory containing images
        output_path: Output video file path
        fps: Frames per second
        max_frames: Maximum number of frames to include
        target_size: Resize images to this size
    """
    image_dir = Path(image_dir)
    
    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(sorted(image_dir.glob(ext)))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    # Limit frames
    image_files = image_files[:max_frames]
    
    print(f"📁 Found {len(image_files)} images")
    print(f"🎬 Creating video: {output_path}")
    print(f"⚙️ Settings: {fps} FPS, {target_size[0]}x{target_size[1]}")
    
    # Read first image to get properties
    first_frame = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
    first_frame = cv2.resize(first_frame, target_size)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        target_size,
        isColor=True  # Convert grayscale to BGR for compatibility
    )
    
    # Write frames
    for img_path in tqdm(image_files, desc="Processing frames"):
        frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(frame, target_size)
        
        # Convert grayscale to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        out.write(frame_bgr)
    
    out.release()
    
    # Get file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    duration = len(image_files) / fps
    
    print(f"\n✅ Video created successfully!")
    print(f"   File: {output_path}")
    print(f"   Frames: {len(image_files)}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Size: {file_size:.2f} MB")


def create_sample_videos():
    """Create multiple sample videos from test dataset"""
    
    # Ensure output directory exists
    output_dir = Path('demo')
    output_dir.mkdir(exist_ok=True)
    
    # Test images directory
    test_images = Path('shared/dataset_v4/test_set/images')
    
    if not test_images.exists():
        print(f"❌ Test images not found at: {test_images}")
        print("   Please check the path or run dataset preparation first")
        return
    
    print("="*60)
    print("Creating Sample Videos for Demo")
    print("="*60)
    print()
    
    # Create multiple videos with different settings
    configs = [
        {
            'name': 'sample_video_short.mp4',
            'max_frames': 30,
            'fps': 10,
            'description': 'Short demo (3 seconds, 30 frames)'
        },
        {
            'name': 'sample_video_medium.mp4',
            'max_frames': 100,
            'fps': 15,
            'description': 'Medium demo (6.7 seconds, 100 frames)'
        },
        {
            'name': 'sample_video_long.mp4',
            'max_frames': 200,
            'fps': 20,
            'description': 'Long demo (10 seconds, 200 frames)'
        }
    ]
    
    for config in configs:
        print(f"\n🎥 {config['description']}")
        print("-" * 60)
        
        output_path = output_dir / config['name']
        
        create_video_from_images(
            image_dir=test_images,
            output_path=output_path,
            fps=config['fps'],
            max_frames=config['max_frames']
        )
    
    print("\n" + "="*60)
    print("✅ All sample videos created!")
    print("="*60)
    print("\n📂 Videos saved in: demo/")
    print("   - sample_video_short.mp4  (quick test)")
    print("   - sample_video_medium.mp4 (recommended)")
    print("   - sample_video_long.mp4   (full demo)")
    print("\n💡 Use these videos in the 'Real-Time Video Stream' tab!")


if __name__ == '__main__':
    create_sample_videos()
