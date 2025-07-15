import React, { useState } from 'react';

function SingleFeedback({ onPredict, predictionResult, isLoading }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) { // Ensure text is not just whitespace
      onPredict(text);
    }
  };

  return (
    <div className="card w-full"> {/* Using global 'card' class, 'w-full' ensures it takes full width of its parent */}
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Analyze Single Feedback</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="feedbackText" className="block text-lg font-medium text-gray-700 mb-2">
            Enter Feedback Text:
          </label>
          <textarea
            id="feedbackText"
            className="input-field min-h-[120px] resize-y" /* Using global input-field class */
            rows="5"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g., This product is fantastic, highly recommend it!"
            disabled={isLoading} // Disable input when loading
          ></textarea>
        </div>
        <button
          type="submit"
          className={`w-full btn-primary ${ /* Using global btn-primary class */
            isLoading ? 'opacity-70 cursor-not-allowed' : ''
          }`}
          disabled={isLoading} // Disable button when loading
        >
          {isLoading ? 'Analyzing...' : 'Analyze Feedback'}
        </button>
      </form>

      {/* Display Prediction Result */}
      {predictionResult && (
        <div className="result-box"> {/* Using global result-box class */}
          <h3 className="text-xl font-semibold text-blue-800 mb-4">Analysis Result:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-gray-700">
            <p className="text-lg">
              <span className="result-label">Authenticity:</span>{' '}
              <span className={`result-value ${
                predictionResult.authenticity === 'Genuine' ? 'text-green' : 'text-red' // Using global text-green/red
              }`}>
                {predictionResult.authenticity}
              </span>
              <span className="text-sm text-gray-500 ml-2">({(predictionResult.authenticity_confidence * 100).toFixed(2)}% confidence)</span>
            </p>
            <p className="text-lg">
              <span className="result-label">Sentiment:</span>{' '}
              <span className={`result-value ${
                predictionResult.sentiment === 'positive' ? 'text-green' : // Using global text-green
                predictionResult.sentiment === 'negative' ? 'text-red' : 'text-yellow' // Using global text-red/yellow
              }`}>
                {predictionResult.sentiment}
              </span>
              <span className="text-sm text-gray-500 ml-2">({(predictionResult.sentiment_confidence * 100).toFixed(2)}% confidence)</span>
            </p>
          </div>
          <p className="mt-4 text-base text-gray-600 italic">
            Original Text: "{predictionResult.feedback_text}"
          </p>
        </div>
      )}
    </div>
  );
}

export default SingleFeedback;
