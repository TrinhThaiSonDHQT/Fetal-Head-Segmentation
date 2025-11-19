import { useState, useEffect, useRef } from 'react';
import { Play, Square } from 'lucide-react';
import Button from './ui/Button.js';
import { Card, CardHeader, CardContent } from './ui/Card.js';
import type { StreamEvent } from '../types/api.js';

export default function StreamDemo() {
  const [streaming, setStreaming] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<StreamEvent | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const startStream = () => {
    const eventSource = new EventSource('/api/stream');
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      const data: StreamEvent = JSON.parse(event.data);
      setCurrentFrame(data);

      if (data.type === 'complete') {
        stopStream();
      }
    };

    eventSource.onerror = () => {
      stopStream();
    };

    setStreaming(true);
  };

  const stopStream = () => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    setStreaming(false);
  };

  useEffect(() => {
    return () => stopStream();
  }, []);

  return (
    <Card>
      <CardHeader>
        <h2 className="text-2xl font-bold">Real-Time Scanning Demo</h2>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex gap-2">
            <Button onClick={startStream} disabled={streaming}>
              <Play className="mr-2 h-4 w-4" />
              Start Demo
            </Button>
            <Button
              onClick={stopStream}
              disabled={!streaming}
              variant="outline"
            >
              <Square className="mr-2 h-4 w-4" />
              Stop
            </Button>
          </div>

          {currentFrame && currentFrame.type === 'frame' && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h3 className="font-semibold mb-2">Live Feed</h3>
                  <img
                    src={`data:image/png;base64,${currentFrame.original}`}
                    alt="Live"
                    className="w-full rounded-md"
                  />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Live Segmentation</h3>
                  <img
                    src={`data:image/png;base64,${currentFrame.segmentation}`}
                    alt="Seg"
                    className="w-full rounded-md"
                  />
                </div>
              </div>
              
              <div className="p-3 bg-blue-50 rounded-md">
                <div className="grid grid-cols-2 gap-4 text-sm text-gray-700">
                  <div>
                    <span className="font-semibold">Frame:</span> {currentFrame.frame_number}
                    {currentFrame.total_frames && ` / ${currentFrame.total_frames}`}
                  </div>
                  {currentFrame.confidence && (
                    <div>
                      <span className="font-semibold">Confidence:</span> {currentFrame.confidence}%
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
