import React, { useState, useEffect, useCallback } from 'react';

// Import components - Using your preferred component names
import SingleFeedback from './components/SingleFeedback.jsx';
import BatchFeedback from './components/BatchFeedback.jsx';
import AnalyticsReport from './components/AnalyticsReport.jsx';

function App() {
  const [view, setView] = useState('single'); // 'single', 'batch', 'report'
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'online', 'offline'
  const [message, setMessage] = useState(''); // General message display
  const [messageType, setMessageType] = useState(''); // 'success', 'error', 'info'
  const [singlePredictionResult, setSinglePredictionResult] = useState(null); // State for single prediction result
  const [batchPredictionResults, setBatchPredictionResults] = useState(null); // State for batch prediction results
  const [loading, setLoading] = useState(false); // Loading state for API calls
  const [error, setError] = useState(null); // Error state for API calls

  const API_BASE_URL = 'http://127.0.0.1:5000'; // Replace with your backend URL if deployed

  // Function to display messages - Wrapped with useCallback
  const showMessage = useCallback((msg, type) => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => {
      setMessage('');
      setMessageType('');
    }, 5000); // Message disappears after 5 seconds
  }, []);

  // Function to check backend status
  const checkBackendStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      if (response.ok) {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
        showMessage('Backend is offline. Please start the Flask server.', 'error');
      }
    } catch (error) {
      console.error("Error checking backend status:", error);
      setBackendStatus('offline');
      showMessage('Backend is offline. Please start the Flask server.', 'error');
    }
  };

  // Function to handle single feedback prediction
  const handleSinglePredict = async (text) => {
    setLoading(true);
    setError(null);
    showMessage('Analyzing feedback...', 'info'); // Show processing message
    setSinglePredictionResult(null); // Clear previous result
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ feedback_text: text }), // Ensure key matches backend
      });
      const data = await response.json();
      if (response.ok) {
        setSinglePredictionResult(data);
        showMessage('Prediction successful!', 'success');
      } else {
        throw new Error(data.error || 'Prediction failed.');
      }
    } catch (err) {
      console.error("Error during single prediction:", err);
      setError(err.message);
      showMessage(`Error: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Function to handle batch feedback prediction (CSV upload)
  const handleBatchPredict = async (file) => {
    setLoading(true);
    setError(null);
    showMessage('Processing CSV...', 'info'); // Show processing message
    setBatchPredictionResults(null); // Clear previous results
    const formData = new FormData();
    formData.append('file', file); // 'file' must match the key Flask expects

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData, // No 'Content-Type' header needed for FormData
      });
      const data = await response.json();
      if (response.ok) {
        setBatchPredictionResults(data.predictions);
        showMessage(data.message || 'Batch prediction successful!', 'success');
      } else {
        throw new Error(data.error || 'Batch prediction failed.');
      }
    } catch (err) {
      console.error("Error during batch prediction:", err);
      setError(err.message);
      showMessage(`Error: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Function to trigger model retraining
  const handleRetrainModels = async () => {
    setLoading(true);
    setError(null);
    showMessage('Retraining models...', 'info');
    try {
      const response = await fetch(`${API_BASE_URL}/retrain`, {
        method: 'POST',
      });
      const data = await response.json();
      if (response.ok) {
        showMessage(data.message || 'Models retrained successfully!', 'success');
      } else {
        showMessage(data.error || 'Failed to retrain models.', 'error');
      }
    } catch (error) {
      console.error("Error during retraining:", error);
      showMessage('Network error during retraining.', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 60000); // Check every minute
    return () => clearInterval(interval);
  }, [showMessage]);

  return (
    // The outermost div now only applies the container class and responsive max-width.
    // Overall page centering and background are handled by #root in index.css.
    <div className="container max-w-6xl lg:max-w-7xl p-4 sm:p-6 lg:p-8">
      {/* Header with custom class for global styles */}
      <div className="header">
        <h1 className="text-3xl sm:text-4xl font-extrabold mb-2">Feedback Authenticity & Sentiment Analysis</h1>
        <p className="text-base sm:text-lg opacity-90">AI-powered system to classify feedback and gain insights.</p>
        <div className={`mt-4 text-sm font-semibold p-2 rounded-md ${backendStatus === 'online' ? 'bg-green-500' : 'bg-red-500'}`}>
          Backend Status: {backendStatus.toUpperCase()}
        </div>
      </div>

      {/* Global Message Display */}
      {message && (
        <div className={`p-3 text-center text-white ${messageType === 'success' ? 'bg-green-500' : messageType === 'error' ? 'bg-red-500' : 'bg-blue-500'}`}>
          {message}
        </div>
      )}

      {/* Navigation */}
      <nav className="p-4 sm:p-6 bg-white flex flex-wrap justify-center gap-3 sm:gap-4 border-b border-gray-200">
        <button
          onClick={() => setView('single')}
          // Using nav-button class from index.css and specific Tailwind classes for active/hover states
          className={`nav-button px-6 py-3 sm:px-8 sm:py-4 rounded-full text-lg ${
            view === 'single' ? 'bg-blue-700 text-white shadow-lg transform -translate-y-1' : 'bg-blue-100 text-blue-800 hover:bg-blue-200 shadow-md'
          }`}
        >
          Single Feedback
        </button>
        <button
          onClick={() => setView('batch')}
          className={`nav-button px-6 py-3 sm:px-8 sm:py-4 rounded-full text-lg ${
            view === 'batch' ? 'bg-blue-700 text-white shadow-lg transform -translate-y-1' : 'bg-blue-100 text-blue-800 hover:bg-blue-200 shadow-md'
          }`}
        >
          Batch Feedback (CSV)
        </button>
        <button
          onClick={() => setView('report')}
          className={`nav-button px-6 py-3 sm:px-8 sm:py-4 rounded-full text-lg ${
            view === 'report' ? 'bg-blue-700 text-white shadow-lg transform -translate-y-1' : 'bg-blue-100 text-blue-800 hover:bg-blue-200 shadow-md'
          }`}
        >
          Analytics Report
        </button>
        <button
          onClick={handleRetrainModels}
          className={`nav-button px-6 py-3 sm:px-8 sm:py-4 rounded-full text-lg ${
            loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 text-white hover:bg-green-700 shadow-lg'
          }`}
          disabled={loading}
        >
          {loading ? 'Retraining...' : 'Retrain Models'}
        </button>
      </nav>

      {/* Content Area - Conditional Rendering */}
      <main className="content-section w-full"> {/* Ensured main takes full width to allow child components to expand */}
        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center z-50">
            <div className="flex flex-col items-center text-white text-xl">
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500 mb-4"></div>
              Processing...
            </div>
          </div>
        )}

        {/* Error Message Display */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Error!</strong>
            <span className="block sm:inline"> {error}</span>
          </div>
        )}

        {view === 'single' && (
          <SingleFeedback
            onPredict={handleSinglePredict}
            predictionResult={singlePredictionResult}
            isLoading={loading}
          />
        )}
        {view === 'batch' && (
          <BatchFeedback
            onPredict={handleBatchPredict}
            predictionResults={batchPredictionResults}
            isLoading={loading}
          />
        )}
        {view === 'report' && (
          <AnalyticsReport
            API_BASE_URL={API_BASE_URL}
          />
        )}
      </main>
    </div>
  );
}

export default App;