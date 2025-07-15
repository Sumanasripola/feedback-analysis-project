import React, { useState } from 'react';

function BatchFeedback({ onPredict, predictionResults, isLoading, showMessage }) {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      // Basic validation for CSV
      if (selectedFile.type === 'text/csv' || selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
        setFileName(selectedFile.name);
        showMessage(`File selected: ${selectedFile.name}`, 'info');
      } else {
        showMessage('Please select a CSV file.', 'error');
        setFile(null);
        setFileName('');
      }
    } else {
      setFile(null);
      setFileName('');
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file) {
      onPredict(file);
    } else {
      showMessage('Please select a file to upload.', 'error');
    }
  };

  return (
    <div className="card w-full"> {/* Using global 'card' class, 'w-full' ensures it takes full width */}
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Analyze Batch Feedback (CSV)</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="csvFile" className="block text-lg font-medium text-gray-700 mb-2">
            Upload CSV File:
            <span className="text-sm text-gray-500 ml-2">(must contain a 'feedback_text' column)</span>
          </label>
          {/* Enhanced Upload Area Styling */}
          <div className="mt-1 flex flex-col items-center justify-center px-6 pt-8 pb-8 border-2 border-dashed border-blue-300 rounded-lg bg-blue-50 hover:bg-blue-100 transition-colors duration-200 cursor-pointer">
            <div className="space-y-2 text-center">
              {/* Larger Icon */}
              <svg
                className="mx-auto h-16 w-16 text-blue-500"
                stroke="currentColor"
                fill="none"
                viewBox="0 0 48 48"
                aria-hidden="true"
              >
                <path
                  d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex text-sm text-gray-600 justify-center">
                <label
                  htmlFor="file-upload"
                  // Applied btn-primary styles to the label to make it look like a button
                  className={`btn-primary relative cursor-pointer bg-blue-600 text-white hover:bg-blue-700 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500 ${isLoading ? 'opacity-70 cursor-not-allowed' : ''}`}
                >
                  <span>Upload a file</span>
                  <input
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".csv"
                    onChange={handleFileChange}
                    disabled={isLoading}
                  />
                </label>
                <p className="pl-1 pt-3">or drag and drop</p> {/* Adjusted padding-top for alignment */}
              </div>
              <p className="text-xs text-gray-500">CSV files only</p>
              {fileName && <p className="text-sm font-medium text-blue-800 mt-2">Selected: {fileName}</p>}
            </div>
          </div>
        </div>
        <button
          type="submit"
          className={`w-full btn-primary ${ /* Using global btn-primary class */
            isLoading || !file ? 'opacity-70 cursor-not-allowed' : ''
          }`}
          disabled={isLoading || !file}
        >
          {isLoading ? 'Processing...' : 'Process CSV'}
        </button>
      </form>

      {/* Display Prediction Results */}
      {predictionResults && predictionResults.length > 0 && (
        <div className="result-box max-h-96 overflow-y-auto"> {/* Applied 'result-box' class */}
          <h3 className="text-xl font-semibold text-blue-800 mb-4">Batch Analysis Summary:</h3>
          <p className="text-lg text-gray-700 mb-4">Processed {predictionResults.length} entries.</p>
          <div className="space-y-4">
            {predictionResults.map((result, index) => ( // Display ALL results
              <div key={index} className="p-4 bg-white rounded-md shadow-sm border border-gray-200">
                <p className="text-base font-medium text-gray-800 truncate">
                  <span className="result-label">Text:</span> {result.feedback_text} {/* Applied 'result-label' class */}
                </p>
                <p className="text-sm text-gray-700 mt-1">
                  <span className="result-label">Authenticity:</span>{' '} {/* Applied 'result-label' class */}
                  <span className={`result-value ${
                    result.authenticity === 'Genuine' ? 'text-green' : 'text-red' // Using global text-green/red classes
                  }`}>
                    {result.authenticity}
                  </span>{' '}
                  ({(result.authenticity_confidence * 100).toFixed(2)}%)
                </p>
                <p className="text-sm text-gray-700">
                  <span className="result-label">Sentiment:</span>{' '} {/* Applied 'result-label' class */}
                  <span className={`result-value ${
                    result.sentiment === 'positive' ? 'text-green' :
                    result.sentiment === 'negative' ? 'text-red' : 'text-yellow'
                  }`}> {/* Using global text-green/red/yellow classes */}
                    {result.sentiment}
                  </span>{' '}
                  ({(result.sentiment_confidence * 100).toFixed(2)}%)
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default BatchFeedback;
