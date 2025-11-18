import { useState, useRef } from 'react';
import { Upload, Loader2 } from 'lucide-react';
import Button from './ui/Button.js';
import { Card, CardHeader, CardContent } from './ui/Card.js';
import { useUploadImageMutation } from '../store/api/segmentationApi.js';

export default function ImageUpload() {
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadImage, { data: result, isLoading, error }] =
    useUploadImageMutation();

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

  return (
    <Card>
      <CardHeader>
        <h2 className="text-2xl font-bold">Upload Ultrasound Image</h2>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="hidden"
          />
          <Button
            onClick={() => fileInputRef.current?.click()}
            variant="outline"
            className="w-full"
          >
            <Upload className="mr-2 h-4 w-4" />
            {file ? file.name : 'Choose Image'}
          </Button>
          <Button
            onClick={handleUpload}
            disabled={!file || isLoading}
            className="w-full"
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

          {result && (
            <div className="grid grid-cols-2 gap-4 mt-6">
              <div>
                <h3 className="font-semibold mb-2">Original</h3>
                <img
                  src={`data:image/png;base64,${result.original}`}
                  alt="Original"
                  className="w-full rounded-md"
                />
              </div>
              <div>
                <h3 className="font-semibold mb-2">Segmentation</h3>
                <img
                  src={`data:image/png;base64,${result.segmentation}`}
                  alt="Result"
                  className="w-full rounded-md"
                />
              </div>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-red-800 text-sm">
                Upload failed. Please try again.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
