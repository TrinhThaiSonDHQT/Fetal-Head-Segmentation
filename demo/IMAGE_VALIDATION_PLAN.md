# Image Validation & Quality Control Implementation Plan

## Overview

Implement a **Hybrid Validation Approach** to handle non-ultrasound images gracefully and improve user experience. This plan adds intelligent validation at multiple stages to ensure only valid fetal head ultrasound images are processed, or users are appropriately warned.

---

## Phase 1: Update TypeScript Interfaces

### Task 1.1: Extend API Response Types

**Goal:** Add validation-related fields to the API response interface.

**Actions:**

1. Update `frontend/src/types/api.ts`:

   ```typescript
   export interface UploadResponse {
     success: boolean;
     original: string; // Base64 encoded
     segmentation: string; // Base64 encoded
     inference_time: number; // milliseconds
     
     // New validation fields
     is_valid_ultrasound: boolean; // Is this likely an ultrasound image?
     confidence_score: number; // Model confidence (0-1)
     quality_metrics: QualityMetrics; // Segmentation quality analysis
     warnings: string[]; // Array of warning messages
     
     error?: string;
   }

   export interface QualityMetrics {
     mask_area_ratio: number; // Ratio of mask to image area
     mask_circularity: number; // How circular/elliptical is the mask (0-1)
     edge_sharpness: number; // Sharpness of mask edges
     is_valid_shape: boolean; // Is the shape reasonable for fetal head?
   }

   export interface StreamEvent {
     type: 'frame' | 'complete' | 'error';
     original?: string;
     segmentation?: string;
     frame_number?: number;
     
     // New validation fields for stream
     confidence_score?: number;
     warnings?: string[];
     
     message?: string;
   }
   ```

**Verification:**

- [ ] Types updated in `api.ts`
- [ ] TypeScript compilation passes
- [ ] No type errors in existing code

---

## Phase 2: Backend - Quality Metrics Module

### Task 2.1: Create Quality Analysis Utilities

**Goal:** Implement functions to analyze segmentation mask quality.

**Actions:**

1. Create `backend/quality_checker.py`:

   ```python
   import numpy as np
   import cv2
   from typing import Dict, Tuple

   class QualityChecker:
       """Analyzes segmentation quality to detect invalid/poor results."""
       
       def __init__(self):
           # Thresholds based on fetal head characteristics
           self.MIN_AREA_RATIO = 0.05  # Mask should be at least 5% of image
           self.MAX_AREA_RATIO = 0.60  # Mask shouldn't exceed 60% of image
           self.MIN_CIRCULARITY = 0.60  # Fetal head is roughly circular
           self.MIN_EDGE_SHARPNESS = 0.3  # Edges should be reasonably sharp
       
       def analyze_mask(self, mask: np.ndarray) -> Dict:
           """
           Analyze segmentation mask quality.
           
           Args:
               mask: Binary mask (0-255 or 0-1)
           
           Returns:
               Dictionary with quality metrics
           """
           # Normalize mask to 0-255
           if mask.max() <= 1:
               mask = (mask * 255).astype(np.uint8)
           
           # Calculate metrics
           area_ratio = self._calculate_area_ratio(mask)
           circularity = self._calculate_circularity(mask)
           edge_sharpness = self._calculate_edge_sharpness(mask)
           
           # Determine if shape is valid
           is_valid_shape = self._is_valid_fetal_head_shape(
               area_ratio, circularity, edge_sharpness
           )
           
           return {
               'mask_area_ratio': round(area_ratio, 4),
               'mask_circularity': round(circularity, 4),
               'edge_sharpness': round(edge_sharpness, 4),
               'is_valid_shape': is_valid_shape
           }
       
       def _calculate_area_ratio(self, mask: np.ndarray) -> float:
           """Calculate ratio of mask area to total image area."""
           total_pixels = mask.shape[0] * mask.shape[1]
           mask_pixels = np.count_nonzero(mask > 127)
           return mask_pixels / total_pixels
       
       def _calculate_circularity(self, mask: np.ndarray) -> float:
           """
           Calculate how circular/elliptical the mask is.
           Circularity = 4π × Area / Perimeter²
           Perfect circle = 1.0, lower values = less circular
           """
           # Find contours
           contours, _ = cv2.findContours(
               mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
           )
           
           if not contours:
               return 0.0
           
           # Get largest contour (main mask)
           main_contour = max(contours, key=cv2.contourArea)
           area = cv2.contourArea(main_contour)
           perimeter = cv2.arcLength(main_contour, True)
           
           if perimeter == 0:
               return 0.0
           
           circularity = (4 * np.pi * area) / (perimeter ** 2)
           return min(circularity, 1.0)  # Cap at 1.0
       
       def _calculate_edge_sharpness(self, mask: np.ndarray) -> float:
           """
           Calculate edge sharpness using gradient magnitude.
           Sharp edges = higher values
           """
           # Calculate gradients
           grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
           grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
           gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
           
           # Normalize to 0-1
           if gradient_magnitude.max() > 0:
               sharpness = gradient_magnitude.mean() / gradient_magnitude.max()
           else:
               sharpness = 0.0
           
           return sharpness
       
       def _is_valid_fetal_head_shape(
           self, area_ratio: float, circularity: float, edge_sharpness: float
       ) -> bool:
           """Determine if metrics indicate a valid fetal head segmentation."""
           return (
               self.MIN_AREA_RATIO <= area_ratio <= self.MAX_AREA_RATIO and
               circularity >= self.MIN_CIRCULARITY and
               edge_sharpness >= self.MIN_EDGE_SHARPNESS
           )
       
       def generate_warnings(self, metrics: Dict) -> list:
           """Generate human-readable warnings based on quality metrics."""
           warnings = []
           
           area_ratio = metrics['mask_area_ratio']
           circularity = metrics['mask_circularity']
           edge_sharpness = metrics['edge_sharpness']
           
           if area_ratio < self.MIN_AREA_RATIO:
               warnings.append(
                   "Segmentation area is very small. This may not be a valid ultrasound image."
               )
           elif area_ratio > self.MAX_AREA_RATIO:
               warnings.append(
                   "Segmentation area is unusually large. Image quality may be poor."
               )
           
           if circularity < self.MIN_CIRCULARITY:
               warnings.append(
                   "Detected shape is not circular/elliptical. "
                   "This may not be a fetal head ultrasound."
               )
           
           if edge_sharpness < self.MIN_EDGE_SHARPNESS:
               warnings.append(
                   "Segmentation edges are unclear. Image may be low quality or incorrect."
               )
           
           if not metrics['is_valid_shape']:
               warnings.append(
                   "⚠️ The uploaded image may not be a fetal head ultrasound. "
                   "Results may be inaccurate."
               )
           
           return warnings
   ```

**Verification:**

- [ ] `quality_checker.py` created
- [ ] All methods implemented
- [ ] Imports work correctly
- [ ] Test with sample mask to verify calculations

---

## Phase 3: Backend - Ultrasound Image Classifier

### Task 3.1: Create Lightweight Ultrasound Detector

**Goal:** Add a simple classifier to detect if input is an ultrasound image.

**Actions:**

1. Create `backend/ultrasound_detector.py`:

   ```python
   import numpy as np
   import cv2
   from typing import Tuple

   class UltrasoundDetector:
       """
       Lightweight detector to identify if an image is likely an ultrasound.
       Uses heuristics based on ultrasound image characteristics.
       """
       
       def __init__(self):
           # Ultrasound images typically have these characteristics
           self.confidence_threshold = 0.5
       
       def is_ultrasound(self, image: np.ndarray) -> Tuple[bool, float]:
           """
           Determine if image is likely an ultrasound.
           
           Args:
               image: Input image (RGB or grayscale)
           
           Returns:
               (is_ultrasound, confidence_score)
           """
           # Convert to grayscale if needed
           if len(image.shape) == 3:
               gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
           else:
               gray = image
           
           # Calculate features
           features = self._extract_features(gray)
           
           # Compute confidence score
           confidence = self._compute_confidence(features)
           
           is_us = confidence >= self.confidence_threshold
           
           return is_us, confidence
       
       def _extract_features(self, gray_image: np.ndarray) -> dict:
           """Extract features characteristic of ultrasound images."""
           h, w = gray_image.shape
           
           # Feature 1: Grayscale distribution (ultrasounds are predominantly dark/mid-tones)
           hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
           hist = hist.flatten() / hist.sum()  # Normalize
           
           # Check if most pixels are in dark/mid range (0-180)
           dark_mid_ratio = hist[0:180].sum()
           
           # Feature 2: Contrast (ultrasounds have moderate contrast)
           contrast = gray_image.std() / 255.0
           
           # Feature 3: Edge density (ultrasounds have specific edge patterns)
           edges = cv2.Canny(gray_image, 50, 150)
           edge_density = np.count_nonzero(edges) / (h * w)
           
           # Feature 4: Corner regions are often black in ultrasounds (cone shape)
           corner_darkness = self._check_corner_darkness(gray_image)
           
           # Feature 5: Texture uniformity (speckle noise pattern)
           texture_score = self._calculate_texture_uniformity(gray_image)
           
           return {
               'dark_mid_ratio': dark_mid_ratio,
               'contrast': contrast,
               'edge_density': edge_density,
               'corner_darkness': corner_darkness,
               'texture_score': texture_score
           }
       
       def _check_corner_darkness(self, gray_image: np.ndarray) -> float:
           """Check if corner regions are predominantly dark (typical in ultrasound)."""
           h, w = gray_image.shape
           corner_size = min(h, w) // 6
           
           # Sample corner regions
           corners = [
               gray_image[0:corner_size, 0:corner_size],  # Top-left
               gray_image[0:corner_size, -corner_size:],  # Top-right
               gray_image[-corner_size:, 0:corner_size],  # Bottom-left
               gray_image[-corner_size:, -corner_size:],  # Bottom-right
           ]
           
           # Calculate mean darkness (lower = darker)
           darkness_scores = [1.0 - (corner.mean() / 255.0) for corner in corners]
           return np.mean(darkness_scores)
       
       def _calculate_texture_uniformity(self, gray_image: np.ndarray) -> float:
           """Calculate texture uniformity (ultrasounds have speckle pattern)."""
           # Use local binary patterns or simple variance measure
           # Higher variance in local patches = more texture
           kernel_size = 8
           h, w = gray_image.shape
           
           variances = []
           for i in range(0, h - kernel_size, kernel_size):
               for j in range(0, w - kernel_size, kernel_size):
                   patch = gray_image[i:i+kernel_size, j:j+kernel_size]
                   variances.append(patch.var())
           
           # Normalize
           texture_score = np.mean(variances) / 255.0 if variances else 0.0
           return texture_score
       
       def _compute_confidence(self, features: dict) -> float:
           """
           Compute confidence score that image is an ultrasound.
           Weighted combination of features.
           """
           score = 0.0
           
           # Dark/mid-tone ratio (ultrasounds are typically 60-85% dark/mid)
           if 0.6 <= features['dark_mid_ratio'] <= 0.95:
               score += 0.25
           
           # Moderate contrast (ultrasounds: 0.15-0.35)
           if 0.10 <= features['contrast'] <= 0.40:
               score += 0.20
           
           # Edge density (ultrasounds: 0.05-0.20)
           if 0.03 <= features['edge_density'] <= 0.25:
               score += 0.20
           
           # Dark corners (ultrasounds: > 0.4)
           if features['corner_darkness'] > 0.3:
               score += 0.20
           
           # Texture (ultrasounds have speckle: 0.05-0.25)
           if 0.03 <= features['texture_score'] <= 0.30:
               score += 0.15
           
           return min(score, 1.0)
   ```

**Verification:**

- [ ] `ultrasound_detector.py` created
- [ ] Feature extraction works
- [ ] Test with ultrasound image → high confidence
- [ ] Test with random photo → low confidence

---

## Phase 4: Backend - Update Inference Pipeline

### Task 4.1: Integrate Validation into Inference

**Goal:** Add validation steps to the inference pipeline.

**Actions:**

1. Update `backend/inference.py`:

   ```python
   import numpy as np
   import cv2
   import torch
   from typing import Dict, Tuple
   from quality_checker import QualityChecker
   from ultrasound_detector import UltrasoundDetector

   class InferenceEngine:
       def __init__(self, model, device):
           self.model = model
           self.device = device
           self.quality_checker = QualityChecker()
           self.us_detector = UltrasoundDetector()
       
       def process_image(self, image: np.ndarray) -> Dict:
           """
           Complete inference pipeline with validation.
           
           Args:
               image: Input image (RGB/grayscale)
           
           Returns:
               Dictionary with results and validation info
           """
           # Step 1: Check if image is ultrasound
           is_ultrasound, us_confidence = self.us_detector.is_ultrasound(image)
           
           # Step 2: Preprocess and run inference
           input_tensor = self.preprocess(image)
           mask = self.run_inference(input_tensor)
           
           # Step 3: Analyze mask quality
           quality_metrics = self.quality_checker.analyze_mask(mask)
           
           # Step 4: Generate warnings
           warnings = []
           
           if not is_ultrasound:
               warnings.append(
                   "⚠️ This image may not be an ultrasound. "
                   f"Confidence: {us_confidence*100:.1f}%"
               )
           
           quality_warnings = self.quality_checker.generate_warnings(quality_metrics)
           warnings.extend(quality_warnings)
           
           return {
               'mask': mask,
               'is_valid_ultrasound': is_ultrasound,
               'confidence_score': us_confidence,
               'quality_metrics': quality_metrics,
               'warnings': warnings
           }
       
       def preprocess(self, image: np.ndarray) -> torch.Tensor:
           """Preprocess image for model input."""
           # Implementation as before
           pass
       
       def run_inference(self, input_tensor: torch.Tensor) -> np.ndarray:
           """Run model inference."""
           # Implementation as before
           pass
   ```

**Verification:**

- [ ] `inference.py` updated
- [ ] Integration works correctly
- [ ] All validation steps execute
- [ ] Test with various image types

---

## Phase 5: Backend - Update API Response

### Task 5.1: Modify Upload Endpoint

**Goal:** Return validation data in API response.

**Actions:**

1. Update `backend/app.py` - `/api/upload` endpoint:

   ```python
   @app.route('/api/upload', methods=['POST'])
   def upload_image():
       try:
           # ... existing code to receive file ...
           
           # Process with validation
           result = inference_engine.process_image(image)
           
           # ... existing code to create visualization ...
           
           response = {
               'success': True,
               'original': base64_original,
               'segmentation': base64_segmentation,
               'inference_time': inference_time,
               
               # Add validation data
               'is_valid_ultrasound': result['is_valid_ultrasound'],
               'confidence_score': float(result['confidence_score']),
               'quality_metrics': result['quality_metrics'],
               'warnings': result['warnings']
           }
           
           return jsonify(response)
       
       except Exception as e:
           return jsonify({
               'success': False,
               'error': str(e),
               'warnings': ['An error occurred during processing']
           }), 500
   ```

2. Update `backend/app.py` - `/api/stream` endpoint similarly for consistency

**Verification:**

- [ ] API returns new fields
- [ ] JSON structure matches TypeScript types
- [ ] Error handling preserved
- [ ] Test with Postman/curl

---

## Phase 6: Frontend - UI Alert Components

### Task 6.1: Create Warning/Alert Components

**Goal:** Build UI components to display warnings.

**Actions:**

1. Create `frontend/src/components/ui/Alert.tsx`:

   ```tsx
   import { HTMLAttributes, forwardRef } from 'react';
   import { AlertCircle, CheckCircle, Info, AlertTriangle } from 'lucide-react';
   import { cn } from '@/lib/utils';
   import { cva, type VariantProps } from 'class-variance-authority';

   const alertVariants = cva(
     'relative w-full rounded-lg border p-4 [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg+div]:pl-8',
     {
       variants: {
         variant: {
           default: 'bg-white border-gray-200 text-gray-900',
           warning: 'bg-yellow-50 border-yellow-200 text-yellow-900',
           error: 'bg-red-50 border-red-200 text-red-900',
           success: 'bg-green-50 border-green-200 text-green-900',
           info: 'bg-blue-50 border-blue-200 text-blue-900',
         },
       },
       defaultVariants: {
         variant: 'default',
       },
     }
   );

   export interface AlertProps
     extends HTMLAttributes<HTMLDivElement>,
       VariantProps<typeof alertVariants> {}

   const Alert = forwardRef<HTMLDivElement, AlertProps>(
     ({ className, variant, children, ...props }, ref) => {
       const Icon = {
         default: Info,
         warning: AlertTriangle,
         error: AlertCircle,
         success: CheckCircle,
         info: Info,
       }[variant || 'default'];

       return (
         <div
           ref={ref}
           role="alert"
           className={cn(alertVariants({ variant }), className)}
           {...props}
         >
           <Icon className="h-5 w-5" />
           <div>{children}</div>
         </div>
       );
     }
   );

   const AlertTitle = forwardRef<
     HTMLParagraphElement,
     HTMLAttributes<HTMLHeadingElement>
   >(({ className, ...props }, ref) => (
     <h5
       ref={ref}
       className={cn('mb-1 font-semibold leading-none tracking-tight', className)}
       {...props}
     />
   ));

   const AlertDescription = forwardRef<
     HTMLParagraphElement,
     HTMLAttributes<HTMLParagraphElement>
   >(({ className, ...props }, ref) => (
     <div
       ref={ref}
       className={cn('text-sm [&_p]:leading-relaxed', className)}
       {...props}
     />
   ));

   export { Alert, AlertTitle, AlertDescription };
   ```

2. Create `frontend/src/components/ui/Badge.tsx`:

   ```tsx
   import { HTMLAttributes, forwardRef } from 'react';
   import { cn } from '@/lib/utils';
   import { cva, type VariantProps } from 'class-variance-authority';

   const badgeVariants = cva(
     'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors',
     {
       variants: {
         variant: {
           default: 'bg-primary/10 text-primary',
           success: 'bg-green-100 text-green-800',
           warning: 'bg-yellow-100 text-yellow-800',
           error: 'bg-red-100 text-red-800',
           secondary: 'bg-gray-100 text-gray-800',
         },
       },
       defaultVariants: {
         variant: 'default',
       },
     }
   );

   export interface BadgeProps
     extends HTMLAttributes<HTMLDivElement>,
       VariantProps<typeof badgeVariants> {}

   const Badge = forwardRef<HTMLDivElement, BadgeProps>(
     ({ className, variant, ...props }, ref) => {
       return (
         <div
           ref={ref}
           className={cn(badgeVariants({ variant }), className)}
           {...props}
         />
       );
     }
   );

   export { Badge, badgeVariants };
   ```

**Verification:**

- [ ] Alert component renders
- [ ] Badge component renders
- [ ] All variants display correctly
- [ ] Icons show properly

---

## Phase 7: Frontend - Update ImageUpload Component

### Task 7.1: Display Validation Results

**Goal:** Show warnings and quality metrics to users.

**Actions:**

1. Update `frontend/src/components/ImageUpload.tsx`:

   ```tsx
   import { useState, useRef } from 'react';
   import { Upload, Loader2, CheckCircle, AlertTriangle } from 'lucide-react';
   import Button from './ui/Button';
   import { Card, CardHeader, CardContent } from './ui/Card';
   import { Alert, AlertTitle, AlertDescription } from './ui/Alert';
   import { Badge } from './ui/Badge';
   import { useUploadImageMutation } from '../store/api/segmentationApi';

   export default function ImageUpload() {
     const [file, setFile] = useState<File | null>(null);
     const [preview, setPreview] = useState<string | null>(null);
     const fileInputRef = useRef<HTMLInputElement>(null);
     const [uploadImage, { data: result, isLoading, error }] =
       useUploadImageMutation();

     const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
       const selectedFile = e.target.files?.[0];
       if (!selectedFile) return;

       setFile(selectedFile);

       // Create preview
       const reader = new FileReader();
       reader.onload = (e) => setPreview(e.target?.result as string);
       reader.readAsDataURL(selectedFile);
     };

     const handleUpload = async () => {
       if (!file) return;

       const formData = new FormData();
       formData.append('image', file);

       try {
         await uploadImage(formData).unwrap();
       } catch (err) {
         console.error('Upload failed:', err);
       }
     };

     const getConfidenceBadge = (score: number) => {
       if (score >= 0.7) return <Badge variant="success">High Confidence</Badge>;
       if (score >= 0.4) return <Badge variant="warning">Medium Confidence</Badge>;
       return <Badge variant="error">Low Confidence</Badge>;
     };

     return (
       <Card>
         <CardHeader>
           <h2 className="text-2xl font-bold">Upload Ultrasound Image</h2>
           <p className="text-sm text-gray-600 mt-1">
             Upload a fetal head ultrasound image for segmentation analysis
           </p>
         </CardHeader>
         <CardContent>
           <div className="space-y-4">
             {/* File Input */}
             <input
               ref={fileInputRef}
               type="file"
               accept="image/*"
               onChange={handleFileSelect}
               className="hidden"
             />

             {/* Image Preview */}
             {preview && (
               <div className="relative">
                 <img
                   src={preview}
                   alt="Preview"
                   className="w-full h-48 object-contain bg-gray-50 rounded-lg border"
                 />
               </div>
             )}

             {/* Upload Buttons */}
             <div className="flex gap-2">
               <Button
                 onClick={() => fileInputRef.current?.click()}
                 variant="outline"
                 className="flex-1"
               >
                 <Upload className="mr-2 h-4 w-4" />
                 {file ? file.name : 'Choose Image'}
               </Button>
               <Button
                 onClick={handleUpload}
                 disabled={!file || isLoading}
                 className="flex-1"
               >
                 {isLoading ? (
                   <>
                     <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                     Processing...
                   </>
                 ) : (
                   'Segment Image'
                 )}
               </Button>
             </div>

             {/* Validation Warnings */}
             {result && result.warnings && result.warnings.length > 0 && (
               <Alert variant={result.is_valid_ultrasound ? 'warning' : 'error'}>
                 <AlertTitle>Validation Warnings</AlertTitle>
                 <AlertDescription>
                   <ul className="list-disc list-inside space-y-1 mt-2">
                     {result.warnings.map((warning, idx) => (
                       <li key={idx} className="text-sm">
                         {warning}
                       </li>
                     ))}
                   </ul>
                 </AlertDescription>
               </Alert>
             )}

             {/* Quality Metrics Display */}
             {result && (
               <div className="space-y-4">
                 {/* Confidence Score */}
                 <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                   <span className="text-sm font-medium">Ultrasound Confidence</span>
                   <div className="flex items-center gap-2">
                     <span className="text-sm font-mono">
                       {(result.confidence_score * 100).toFixed(1)}%
                     </span>
                     {getConfidenceBadge(result.confidence_score)}
                   </div>
                 </div>

                 {/* Quality Metrics */}
                 <div className="grid grid-cols-2 gap-3">
                   <div className="p-3 bg-gray-50 rounded-lg">
                     <div className="text-xs text-gray-600 mb-1">Mask Coverage</div>
                     <div className="font-mono text-sm">
                       {(result.quality_metrics.mask_area_ratio * 100).toFixed(1)}%
                     </div>
                   </div>
                   <div className="p-3 bg-gray-50 rounded-lg">
                     <div className="text-xs text-gray-600 mb-1">Shape Quality</div>
                     <div className="font-mono text-sm">
                       {(result.quality_metrics.mask_circularity * 100).toFixed(1)}%
                     </div>
                   </div>
                   <div className="p-3 bg-gray-50 rounded-lg">
                     <div className="text-xs text-gray-600 mb-1">Edge Sharpness</div>
                     <div className="font-mono text-sm">
                       {(result.quality_metrics.edge_sharpness * 100).toFixed(1)}%
                     </div>
                   </div>
                   <div className="p-3 bg-gray-50 rounded-lg">
                     <div className="text-xs text-gray-600 mb-1">Valid Shape</div>
                     <div className="flex items-center gap-1">
                       {result.quality_metrics.is_valid_shape ? (
                         <>
                           <CheckCircle className="h-4 w-4 text-green-600" />
                           <span className="text-sm text-green-600">Yes</span>
                         </>
                       ) : (
                         <>
                           <AlertTriangle className="h-4 w-4 text-yellow-600" />
                           <span className="text-sm text-yellow-600">No</span>
                         </>
                       )}
                     </div>
                   </div>
                 </div>

                 {/* Results */}
                 <div className="grid grid-cols-2 gap-4 mt-6">
                   <div>
                     <h3 className="font-semibold mb-2">Original</h3>
                     <img
                       src={`data:image/png;base64,${result.original}`}
                       alt="Original"
                       className="rounded-lg border"
                     />
                   </div>
                   <div>
                     <h3 className="font-semibold mb-2">Segmentation</h3>
                     <img
                       src={`data:image/png;base64,${result.segmentation}`}
                       alt="Result"
                       className="rounded-lg border"
                     />
                   </div>
                 </div>

                 {/* Inference Time */}
                 <div className="text-xs text-gray-500 text-center">
                   Processed in {result.inference_time}ms
                 </div>
               </div>
             )}

             {/* Error Display */}
             {error && (
               <Alert variant="error">
                 <AlertTitle>Error</AlertTitle>
                 <AlertDescription>
                   Failed to process image. Please try again.
                 </AlertDescription>
               </Alert>
             )}
           </div>
         </CardContent>
       </Card>
     );
   }
   ```

**Verification:**

- [ ] Component renders correctly
- [ ] Preview shows selected image
- [ ] Warnings display when present
- [ ] Quality metrics show correctly
- [ ] Confidence badge reflects score
- [ ] Responsive layout maintained

---

## Phase 8: Frontend - Add Usage Guidelines

### Task 8.1: Create Info/Help Component

**Goal:** Guide users on what images to upload.

**Actions:**

1. Create `frontend/src/components/UsageGuide.tsx`:

   ```tsx
   import { Info, Check, X } from 'lucide-react';
   import { Alert, AlertTitle, AlertDescription } from './ui/Alert';

   export default function UsageGuide() {
     return (
       <Alert variant="info">
         <AlertTitle className="flex items-center gap-2">
           <Info className="h-4 w-4" />
           Image Guidelines
         </AlertTitle>
         <AlertDescription>
           <div className="grid md:grid-cols-2 gap-4 mt-3">
             <div>
               <h4 className="font-semibold text-sm mb-2 flex items-center gap-1">
                 <Check className="h-4 w-4 text-green-600" />
                 Upload These:
               </h4>
               <ul className="text-xs space-y-1 text-gray-700">
                 <li>• Fetal head ultrasound scans</li>
                 <li>• 2D B-mode ultrasound images</li>
                 <li>• Clear head circumference views</li>
                 <li>• Standard ultrasound formats</li>
               </ul>
             </div>
             <div>
               <h4 className="font-semibold text-sm mb-2 flex items-center gap-1">
                 <X className="h-4 w-4 text-red-600" />
                 Avoid These:
               </h4>
               <ul className="text-xs space-y-1 text-gray-700">
                 <li>• Regular photos or screenshots</li>
                 <li>• Non-medical images</li>
                 <li>• 3D/4D ultrasound renders</li>
                 <li>• Other body part scans</li>
               </ul>
             </div>
           </div>
         </AlertDescription>
       </Alert>
     );
   }
   ```

2. Add to `ImageUpload.tsx` at the top:

   ```tsx
   import UsageGuide from './UsageGuide';

   // Inside CardContent, before file input:
   <UsageGuide />
   ```

**Verification:**

- [ ] Guide displays correctly
- [ ] Helpful for users
- [ ] Responsive layout
- [ ] Clear do's and don'ts

---

## Phase 9: Testing & Refinement

### Task 9.1: Comprehensive Testing

**Goal:** Test all validation scenarios.

**Test Cases:**

1. **Valid Ultrasound Images:**

   - Upload fetal head ultrasound → Should show high confidence, no warnings
   - Quality metrics should be in valid range
   - Segmentation should work correctly

2. **Invalid/Non-Ultrasound Images:**

   - Upload regular photo → Should show low confidence, warnings
   - Upload screenshot → Should warn user
   - Upload colorful image → Should detect as non-ultrasound

3. **Edge Cases:**

   - Upload very dark image → Check edge sharpness warning
   - Upload image with no clear object → Check area ratio warning
   - Upload image with rectangular object → Check circularity warning

4. **UI/UX:**
   - Warnings are clear and actionable
   - Metrics display correctly
   - Confidence badge colors make sense
   - Layout doesn't break with long warnings

**Verification:**

- [ ] All test cases pass
- [ ] False positives are minimal
- [ ] False negatives are minimal
- [ ] User experience is smooth
- [ ] Performance is acceptable (<1s per image)

### Task 9.2: Threshold Tuning

**Goal:** Optimize detection thresholds based on real data.

**Actions:**

1. Test with 20+ ultrasound images from dataset
2. Test with 20+ random non-ultrasound images
3. Calculate:
   - True Positive Rate (correctly identified ultrasounds)
   - False Positive Rate (non-ultrasounds marked as valid)
   - False Negative Rate (ultrasounds marked as invalid)
4. Adjust thresholds in `QualityChecker` and `UltrasoundDetector` if needed
5. Document final threshold values

**Target Metrics:**

- True Positive Rate: >90%
- False Positive Rate: <15%
- False Negative Rate: <10%

**Verification:**

- [ ] Threshold testing completed
- [ ] Metrics meet targets
- [ ] Thresholds documented
- [ ] Production-ready

---

## Phase 10: Documentation & Deployment

### Task 10.1: Update Documentation

**Goal:** Document the validation system.

**Actions:**

1. Update `README.md` with validation features
2. Add section explaining quality metrics
3. Document threshold values and their meanings
4. Add troubleshooting guide for common warnings

**Verification:**

- [ ] Documentation complete
- [ ] Clear explanations
- [ ] Examples provided
- [ ] Ready for users

---

## Implementation Checklist

**Phase 1: TypeScript Interfaces**

- [ ] Update `api.ts` with new fields
- [ ] Add `QualityMetrics` interface
- [ ] Update `StreamEvent` interface

**Phase 2: Quality Checker**

- [ ] Create `quality_checker.py`
- [ ] Implement area ratio calculation
- [ ] Implement circularity calculation
- [ ] Implement edge sharpness calculation
- [ ] Implement warning generation
- [ ] Test with sample masks

**Phase 3: Ultrasound Detector**

- [ ] Create `ultrasound_detector.py`
- [ ] Implement feature extraction
- [ ] Implement confidence calculation
- [ ] Test with ultrasound images
- [ ] Test with non-ultrasound images

**Phase 4: Inference Integration**

- [ ] Update `inference.py`
- [ ] Integrate quality checker
- [ ] Integrate ultrasound detector
- [ ] Add warning aggregation
- [ ] Test complete pipeline

**Phase 5: API Update**

- [ ] Update `/api/upload` response
- [ ] Update `/api/stream` response
- [ ] Test API responses
- [ ] Verify JSON structure

**Phase 6: UI Components**

- [ ] Create `Alert.tsx`
- [ ] Create `Badge.tsx`
- [ ] Test all variants
- [ ] Verify styling

**Phase 7: ImageUpload Update**

- [ ] Add image preview
- [ ] Add validation warnings display
- [ ] Add quality metrics display
- [ ] Add confidence badge
- [ ] Test responsive layout

**Phase 8: Usage Guide**

- [ ] Create `UsageGuide.tsx`
- [ ] Add to ImageUpload
- [ ] Test display
- [ ] Verify helpfulness

**Phase 9: Testing**

- [ ] Test with valid ultrasounds
- [ ] Test with invalid images
- [ ] Test edge cases
- [ ] Tune thresholds
- [ ] Verify performance

**Phase 10: Documentation**

- [ ] Update README
- [ ] Document metrics
- [ ] Add troubleshooting
- [ ] Final review

---

## Success Criteria

The validation system is complete when:

- [ ] Backend correctly detects ultrasound vs non-ultrasound images
- [ ] Quality metrics are calculated accurately
- [ ] Warnings are generated appropriately
- [ ] Frontend displays all validation information clearly
- [ ] User experience is improved (no confusion with bad inputs)
- [ ] True positive rate >90% for ultrasound detection
- [ ] False positive rate <15%
- [ ] Performance overhead <100ms per image
- [ ] All components are well-tested
- [ ] Documentation is complete

---

## Execution Timeline

1. **Phase 1-2:** 30-45 minutes (Backend foundation)
2. **Phase 3-4:** 45-60 minutes (Detection & integration)
3. **Phase 5:** 30 minutes (API updates)
4. **Phase 6-7:** 60-90 minutes (Frontend components)
5. **Phase 8:** 20 minutes (Usage guide)
6. **Phase 9:** 60-90 minutes (Testing & tuning)
7. **Phase 10:** 30 minutes (Documentation)

**Total Estimated Time:** 4-6 hours

---

## Next Steps

After reading this plan:

1. Confirm approach is acceptable
2. Start with Phase 1 (TypeScript interfaces)
3. Proceed phase by phase
4. Test incrementally after each phase
5. Request assistance as needed

**Ready to begin implementation!**
