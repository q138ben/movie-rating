'use client';

import { useState } from 'react';

interface MovieFormProps {
  onSubmit: (data: MovieData) => void;
  isLoading: boolean;
}

export interface MovieData {
  title: string;
  description: string;
  tagline: string;
  release_year: number;
  duration: number;
  age_rating: string;
  actors: string[];
  directors: string[];
  studios: string[];
}

export default function MovieForm({ onSubmit, isLoading }: MovieFormProps) {
  const [formData, setFormData] = useState<MovieData>({
    title: '',
    description: '',
    tagline: '',
    release_year: 2024,
    duration: 120,
    age_rating: 'PG-13',
    actors: [''],
    directors: [''],
    studios: ['']
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleArrayChange = (field: keyof MovieData, index: number, value: string) => {
    const newArray = [...formData[field]];
    newArray[index] = value;
    setFormData({ ...formData, [field]: newArray });
  };

  const addArrayField = (field: keyof MovieData) => {
    setFormData({ ...formData, [field]: [...formData[field], ''] });
  };

  const removeArrayField = (field: keyof MovieData, index: number) => {
    const newArray = formData[field].filter((_, i) => i !== index);
    setFormData({ ...formData, [field]: newArray });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6 max-w-2xl mx-auto p-6">
      <div>
        <label className="block text-sm font-medium text-gray-700">Title</label>
        <input
          type="text"
          value={formData.title}
          onChange={(e) => setFormData({ ...formData, title: e.target.value })}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Description</label>
        <textarea
          value={formData.description}
          onChange={(e) => setFormData({ ...formData, description: e.target.value })}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
          rows={4}
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Tagline</label>
        <input
          type="text"
          value={formData.tagline}
          onChange={(e) => setFormData({ ...formData, tagline: e.target.value })}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
          required
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Release Year</label>
          <input
            type="number"
            value={formData.release_year}
            onChange={(e) => setFormData({ ...formData, release_year: parseInt(e.target.value) })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Duration (minutes)</label>
          <input
            type="number"
            value={formData.duration}
            onChange={(e) => setFormData({ ...formData, duration: parseFloat(e.target.value) })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            required
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Age Rating</label>
        <select
          value={formData.age_rating}
          onChange={(e) => setFormData({ ...formData, age_rating: e.target.value })}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
        >
          <option value="G">G</option>
          <option value="PG">PG</option>
          <option value="PG-13">PG-13</option>
          <option value="R">R</option>
          <option value="NC-17">NC-17</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Actors</label>
        {formData.actors.map((actor, index) => (
          <div key={index} className="flex gap-2 mt-2">
            <input
              type="text"
              value={actor}
              onChange={(e) => handleArrayChange('actors', index, e.target.value)}
              className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              placeholder="Actor name"
            />
            <button
              type="button"
              onClick={() => removeArrayField('actors', index)}
              className="px-3 py-2 text-red-600 hover:text-red-800"
            >
              Remove
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={() => addArrayField('actors')}
          className="mt-2 text-indigo-600 hover:text-indigo-800"
        >
          + Add Actor
        </button>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Directors</label>
        {formData.directors.map((director, index) => (
          <div key={index} className="flex gap-2 mt-2">
            <input
              type="text"
              value={director}
              onChange={(e) => handleArrayChange('directors', index, e.target.value)}
              className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              placeholder="Director name"
            />
            <button
              type="button"
              onClick={() => removeArrayField('directors', index)}
              className="px-3 py-2 text-red-600 hover:text-red-800"
            >
              Remove
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={() => addArrayField('directors')}
          className="mt-2 text-indigo-600 hover:text-indigo-800"
        >
          + Add Director
        </button>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Studios</label>
        {formData.studios.map((studio, index) => (
          <div key={index} className="flex gap-2 mt-2">
            <input
              type="text"
              value={studio}
              onChange={(e) => handleArrayChange('studios', index, e.target.value)}
              className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              placeholder="Studio name"
            />
            <button
              type="button"
              onClick={() => removeArrayField('studios', index)}
              className="px-3 py-2 text-red-600 hover:text-red-800"
            >
              Remove
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={() => addArrayField('studios')}
          className="mt-2 text-indigo-600 hover:text-indigo-800"
        >
          + Add Studio
        </button>
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
      >
        {isLoading ? 'Predicting...' : 'Predict Rating'}
      </button>
    </form>
  );
} 