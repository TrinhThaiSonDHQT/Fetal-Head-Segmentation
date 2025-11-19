import { useState, useRef } from 'react';
import { Upload, Loader2, CheckCircle, AlertTriangle } from 'lucide-react';
import Button from './ui/Button.js';
import { Card, CardHeader, CardContent } from './ui/Card.js';
import { Alert, AlertTitle, AlertDescription } from './ui/Alert.js';
import { Badge } from './ui/Badge.js';
import { useUploadImageMutation } from '../store/api/segmentationApi.js';

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
                className="w-full h-96 object-contain bg-gray-50 rounded-lg border p-2"
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
