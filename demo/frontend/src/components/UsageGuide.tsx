import { Info, Check, X } from 'lucide-react';
import { Alert, AlertTitle, AlertDescription } from './ui/Alert.js';

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
            <ul className="text-sm space-y-1 text-gray-700">
              <li>• Fetal head ultrasound scans</li>
              <li>• 2D B-mode ultrasound images</li>
              <li>• Clear head circumference views</li>
              <li>• Common image formats (JPEG, PNG, BMP)</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-sm mb-2 flex items-center gap-1">
              <X className="h-4 w-4 text-red-600" />
              Avoid These:
            </h4>
            <ul className="text-sm space-y-1 text-gray-700">
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
