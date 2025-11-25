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

export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
}
