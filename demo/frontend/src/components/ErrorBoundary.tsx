import React, { Component, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Card, CardHeader, CardContent } from './ui/Card';
import Button from './ui/Button';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

/**
 * Error Boundary Component
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI.
 */
class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(_error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error details for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // You can also log the error to an error reporting service here
    // logErrorToService(error, errorInfo);
  }

  handleReset = () => {
    // Reset the error boundary state
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      // Fallback UI when an error occurs
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <Card className="max-w-2xl w-full">
            <CardHeader>
              <div className="flex items-center gap-3 text-red-600">
                <AlertTriangle className="h-8 w-8" />
                <h2 className="text-2xl font-bold">Something went wrong</h2>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-gray-700">
                  The application encountered an unexpected error. Please try refreshing the page.
                </p>

                {this.state.error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="font-mono text-sm text-red-800">
                      {this.state.error.toString()}
                    </p>
                  </div>
                )}

                {this.state.errorInfo && import.meta.env.DEV && (
                  <details className="bg-gray-100 rounded-lg p-4">
                    <summary className="cursor-pointer font-semibold text-sm text-gray-700 mb-2">
                      Error Details (Development Only)
                    </summary>
                    <pre className="text-xs overflow-x-auto text-gray-600">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </details>
                )}

                <div className="flex gap-3">
                  <Button onClick={this.handleReset} variant="default">
                    Try Again
                  </Button>
                  <Button
                    onClick={() => window.location.reload()}
                    variant="outline"
                  >
                    Refresh Page
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
