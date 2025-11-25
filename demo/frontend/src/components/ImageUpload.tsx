import { useState, useRef } from 'react';
import { Upload, Loader2, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import Button from './ui/Button.js';
import { Card, CardHeader, CardContent } from './ui/Card.js';
import { Alert, AlertTitle, AlertDescription } from './ui/Alert.js';
import { Badge } from './ui/Badge.js';
import UsageGuide from './UsageGuide.js';
import { useUploadImageMutation } from '../store/api/segmentationApi.js';

export default function ImageUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadImage, { data: result, isLoading, error }] =
    useUploadImageMutation();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    // Validate file type
    const validImageTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validImageTypes.includes(selectedFile.type)) {
      setFileError('Please select a valid image file (JPEG, PNG, GIF, BMP, or WebP)');
      setFile(null);
      setPreview(null);
      e.target.value = ''; // Reset file input
      return;
    }

    // Validate file size (16 MB max)
    const maxSize = 16 * 1024 * 1024; // 16 MB
    if (selectedFile.size > maxSize) {
      setFileError('File size exceeds 16 MB. Please select a smaller image.');
      setFile(null);
      setPreview(null);
      e.target.value = '';
      return;
    }

    // Clear any previous errors
    setFileError(null);
    setUploadError(null);
    setFile(selectedFile);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.onerror = () => {
      setFileError('Failed to read file. The file may be corrupted.');
      setFile(null);
      setPreview(null);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    try {
      setUploadError(null);
      await uploadImage(formData).unwrap();
      setRetryCount(0); // Reset retry count on success
    } catch (err: unknown) {
      console.error('Upload failed:', err);
      
      // Handle different error types
      let errorMessage = 'Failed to process image. Please try again.';
      
      if (err && typeof err === 'object') {
        // Network/timeout errors
        if ('status' in err) {
          const status = (err as { status?: number | string }).status;
          if (status === 408) {
            errorMessage = 'Request timeout. The server took too long to respond. Please try again.';
          } else if (status === 413) {
            errorMessage = 'File too large. Maximum size is 16 MB.';
          } else if (status === 500) {
            errorMessage = 'Server error. Please try again later.';
          } else if (status === 400) {
            // Extract backend error message if available
            const data = (err as { data?: { error?: string } }).data;
            if (data?.error) {
              errorMessage = data.error;
            } else {
              errorMessage = 'Invalid image file. Please upload a valid ultrasound image.';
            }
          } else if (!status || status === 'FETCH_ERROR') {
            errorMessage = 'Network error. Please check your connection and try again.';
          }
        }
      }
      
      setUploadError(errorMessage);
      setRetryCount((prev) => prev + 1);
    }
  };

  const handleRetry = () => {
    handleUpload();
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
          {/* Usage Guidelines */}
          <UsageGuide />

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
                className="w-full h-96 object-contain bg-gray-50 rounded-lg border p-2"
              />
            </div>
          )}

          {/* File Type Error */}
          {fileError && (
            <Alert variant="error">
              <AlertTitle>Invalid File Type</AlertTitle>
              <AlertDescription>{fileError}</AlertDescription>
            </Alert>
          )}

          {/* Upload Buttons */}
          <div className="flex gap-2">
            <Button
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
              className="flex-1"
            >
              <Upload className="mr-2 h-4 w-4" />
              Choose Image
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
              <div className="text-sm text-gray-500 text-center">
                Processed in {result.inference_time}ms
              </div>
            </div>
          )}

          {/* Error Display */}
          {(error || uploadError) && (
            <Alert variant="error">
              <AlertTitle>Error Processing Image</AlertTitle>
              <AlertDescription>
                <div className="space-y-3">
                  <p>{uploadError || 'Failed to process image. Please try again.'}</p>
                  
                  {retryCount > 0 && retryCount < 3 && (
                    <div className="flex items-center gap-2">
                      <Button
                        onClick={handleRetry}
                        variant="outline"
                        size="sm"
                        disabled={isLoading}
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                            Retrying...
                          </>
                        ) : (
                          'Retry Upload'
                        )}
                      </Button>
                      <span className="text-xs text-gray-600">
                        Attempt {retryCount} of 3
                      </span>
                    </div>
                  )}
                  
                  {retryCount >= 3 && (
                    <p className="text-xs text-gray-600">
                      Multiple attempts failed. Please check your image file and network connection.
                    </p>
                  )}
                </div>
              </AlertDescription>
            </Alert>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
