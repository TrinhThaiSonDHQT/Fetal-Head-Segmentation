"""
Inference Module for Fetal Head Segmentation

Handles image preprocessing, model inference, and post-processing
for fetal head segmentation from ultrasound images.
"""
import torch
import cv2
import numpy as np
from PIL import Image
import time


class InferenceEngine:
    """
    Handles the complete inference pipeline:
    1. Preprocess input image
    2. Run model inference
    3. Post-process segmentation mask
    """
    
    def __init__(self, model, device):
        """
        Initialize InferenceEngine.
        
        Args:
            model (nn.Module): Loaded PyTorch model
            device (torch.device): Device for inference
        """
        self.model = model
        self.device = device
        self.input_size = (256, 256)  # Model expects 256x256 input
    
    def preprocess(self, image):
        """
        Preprocess image for model input.
        
        Steps:
        1. Convert to grayscale if needed
        2. Resize to 256x256
        3. Normalize to [0, 1]
        4. Convert to tensor (1, 1, 256, 256)
        
        Args:
            image (np.ndarray or PIL.Image): Input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
            tuple: Original image size (height, width)
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Store original size
        original_size = image.shape[:2]  # (height, width)
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to model input size
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor: (H, W) -> (1, 1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, original_size
    
    def run_inference(self, image_tensor):
        """
        Run model inference on preprocessed image.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image (1, 1, 256, 256)
        
        Returns:
            torch.Tensor: Raw model output (1, 1, 256, 256)
            float: Inference time in milliseconds
        """
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(image_tensor)
            
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return output, inference_time
    
    def postprocess(self, output, original_size, threshold=0.5):
        """
        Post-process model output to binary mask.
        
        Steps:
        1. Apply sigmoid activation
        2. Apply threshold
        3. Resize to original image size
        4. Convert to numpy array
        
        Args:
            output (torch.Tensor): Raw model output (1, 1, 256, 256)
            original_size (tuple): Original image size (height, width)
            threshold (float): Threshold for binary mask (default: 0.5)
        
        Returns:
            np.ndarray: Binary mask (H, W) with values 0 or 255
        """
        # Apply sigmoid to get probabilities
        prob_mask = torch.sigmoid(output)
        
        # Apply threshold
        binary_mask = (prob_mask > threshold).float()
        
        # Convert to numpy and remove batch/channel dims: (1, 1, H, W) -> (H, W)
        mask = binary_mask.squeeze().cpu().numpy()
        
        # Resize to original size
        mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Convert to 0-255 range
        mask = (mask * 255).astype(np.uint8)
        
        return mask
    
    def predict(self, image, threshold=0.5):
        """
        Complete inference pipeline.
        
        Args:
            image (np.ndarray or PIL.Image): Input ultrasound image
            threshold (float): Threshold for binary segmentation
        
        Returns:
            dict: {
                'mask': Binary segmentation mask (np.ndarray),
                'inference_time': Time in milliseconds (float)
            }
        """
        # Preprocess
        image_tensor, original_size = self.preprocess(image)
        
        # Inference
        output, inference_time = self.run_inference(image_tensor)
        
        # Post-process
        mask = self.postprocess(output, original_size, threshold)
        
        return {
            'mask': mask,
            'inference_time': inference_time
        }
