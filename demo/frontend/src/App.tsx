import { Activity } from 'lucide-react';
import ImageUpload from './components/ImageUpload.js';
import ErrorBoundary from './components/ErrorBoundary.js';

function App() {
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 py-6">
            <div className="flex items-center gap-3">
              <Activity className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  Fetal Head Segmentation
                </h1>
                <p className="text-gray-600">AI-powered ultrasound analysis</p>
              </div>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 py-8">
          <ImageUpload />
        </main>

        <footer className="mt-16 py-6 text-center text-gray-600 text-sm border-t">
          <p>Fetal Head Segmentation &copy; 2025</p>
        </footer>
      </div>
    </ErrorBoundary>
  );
}

export default App;
