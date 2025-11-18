export interface UploadResponse {
  success: boolean;
  original: string; // Base64 encoded
  segmentation: string; // Base64 encoded
  inference_time: number; // milliseconds
  error?: string;
}

export interface StreamEvent {
  type: 'frame' | 'complete' | 'error';
  original?: string;
  segmentation?: string;
  frame_number?: number;
  message?: string;
}

export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
}
