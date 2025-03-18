'use client';

import { useState } from 'react';
import MovieForm, { MovieData } from '../components/MovieForm';
import Results from '../components/Results';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (data: MovieData) => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const result = await response.json();
      setPrediction(result.predicted_rating);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Movie Rating Predictor
          </h1>
          <p className="text-lg text-gray-600">
            Enter movie details to predict its rating
          </p>
        </div>

        {error && (
          <div className="max-w-2xl mx-auto mb-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600">{error}</p>
          </div>
        )}

        <MovieForm onSubmit={handleSubmit} isLoading={isLoading} />

        {prediction !== null && (
          <div className="mt-8">
            <Results prediction={prediction} confidence="high" />
          </div>
        )}
      </div>
    </main>
  );
}
