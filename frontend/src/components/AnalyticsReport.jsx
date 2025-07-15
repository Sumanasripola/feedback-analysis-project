import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A28DFF', '#8884d8', '#82ca9d']; // Example colors for charts

function AnalyticsReport({ API_BASE_URL, showMessage }) {
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Function to fetch report data from the backend
  const fetchReport = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/report_data`); // Endpoint for report data
      const data = await response.json();
      if (response.ok) {
        setReportData(data);
        if (showMessage) showMessage('Report data loaded successfully!', 'success');
      } else {
        throw new Error(data.error || 'Failed to fetch report data.');
      }
    } catch (err) {
      console.error("Error fetching report data:", err);
      setError(err.message);
      if (showMessage) showMessage(`Error loading report: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReport(); // Fetch report data when component mounts
  }, [API_BASE_URL, showMessage]); // Added API_BASE_URL and showMessage to dependency array

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500"></div>
        <p className="ml-4 text-xl text-gray-600">Loading Report...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">Error!</strong>
        <span className="block sm:inline"> {error}</span>
      </div>
    );
  }

  if (!reportData || reportData.total_feedback === 0) {
    return (
      <div className="text-center p-8 text-gray-600">
        <p className="text-xl font-semibold mb-4">No data available to generate a report.</p>
        <p>Please submit some feedback first using the "Single Feedback" or "Batch Feedback" options.</p>
      </div>
    );
  }

  return (
    <div className="card w-full"> {/* Applied 'card' class for background, border, shadow, and rounded corners */}
      <h2 className="text-3xl font-bold text-blue-700 mb-6 text-center">Feedback Analytics Report</h2>

      <div className="result-box mb-8"> {/* Applied 'result-box' class for the summary container */}
        <h3 className="text-xl font-semibold text-blue-800 mb-2">Summary</h3>
        <p className="text-gray-700 text-base">{reportData.summary}</p>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
          <div className="p-3 bg-white rounded-lg shadow-sm">
            <p className="text-sm text-gray-500">Total Feedback</p>
            <p className="text-2xl font-bold text-blue-600">{reportData.total_feedback}</p>
          </div>
          <div className="p-3 bg-white rounded-lg shadow-sm">
            <p className="text-sm text-gray-500">Genuine Feedback</p>
            <p className="text-2xl font-bold text-green-600">{reportData.genuine_percentage}%</p>
          </div>
          <div className="p-3 bg-white rounded-lg shadow-sm">
            <p className="text-sm text-gray-500">False Feedback</p>
            <p className="text-2xl font-bold text-red-600">{reportData.false_percentage}%</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Authenticity Distribution Chart */}
        <div className="bg-gray-50 p-4 rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">Authenticity Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={reportData.authenticity_distribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={100}
                fill="#8884d8"
                dataKey="count"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {reportData.authenticity_distribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value, name, props) => [`${value} entries`, props.payload.label]} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Sentiment Distribution Chart */}
        <div className="bg-gray-50 p-4 rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">Sentiment Distribution (Genuine Feedback)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={reportData.sentiment_distribution}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#82ca9d" name="Number of Feedback" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Frequent Keywords - Now as a centered table */}
        <div className="bg-gray-50 p-4 rounded-lg shadow-sm border border-gray-200 flex flex-col items-center">
          <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">Frequent Keywords (Genuine Feedback)</h3>
          {reportData.frequent_keywords && reportData.frequent_keywords.length > 0 ? (
            <div className="overflow-x-auto w-full flex justify-center"> {/* Added flex justify-center */}
              <table className="min-w-max bg-white rounded-lg shadow-md text-gray-700 text-center"> {/* Added text-center */}
                <thead>
                  <tr className="bg-blue-100 text-blue-800 uppercase text-sm leading-normal">
                    <th className="py-3 px-6 text-left">Keyword</th>
                    <th className="py-3 px-6 text-center">Count</th>
                  </tr>
                </thead>
                <tbody className="text-gray-600 text-sm font-light">
                  {reportData.frequent_keywords.map((item, index) => (
                    <tr key={index} className="border-b border-gray-200 hover:bg-gray-100">
                      <td className="py-3 px-6 text-left whitespace-nowrap">
                        <span className="font-medium text-lg">{item.word}</span>
                      </td>
                      <td className="py-3 px-6 text-center">
                        <span className="text-blue-600 font-bold text-xl">{item.count}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-600">No frequent keywords found (insufficient genuine text).</p>
          )}
        </div>

        {/* Word Cloud Image */}
        <div className="bg-gray-50 p-4 rounded-lg shadow-sm border border-gray-200 flex flex-col items-center justify-center">
          <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">Word Cloud (Genuine Feedback)</h3>
          {reportData.wordcloud_image ? (
            <img src={reportData.wordcloud_image} alt="Word Cloud" className="max-w-full h-auto rounded-md shadow-md" />
          ) : (
            <p className="text-gray-600">No word cloud generated (insufficient genuine text).</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default AnalyticsReport;