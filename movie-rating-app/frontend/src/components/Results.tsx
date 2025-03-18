'use client';

interface ResultsProps {
  prediction: number;
  confidence: string;
}

export default function Results({ prediction, confidence }: ResultsProps) {
  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">Prediction Results</h2>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-lg font-medium text-gray-700">Predicted Rating:</span>
          <span className="text-3xl font-bold text-indigo-600">{prediction.toFixed(1)}/10</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-lg font-medium text-gray-700">Confidence:</span>
          <span className={`text-lg font-medium ${
            confidence === 'high' ? 'text-green-600' : 
            confidence === 'medium' ? 'text-yellow-600' : 
            'text-red-600'
          }`}>
            {confidence.charAt(0).toUpperCase() + confidence.slice(1)}
          </span>
        </div>
      </div>
    </div>
  );
} 