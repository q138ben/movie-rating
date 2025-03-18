# Movie Rating Predictor

A full-stack application that predicts movie ratings using machine learning. The application consists of a FastAPI backend and a Next.js frontend.

## Project Structure

```
movie-rating-app/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main FastAPI application
│   ├── model.py            # ML model definition
│   ├── feature_engineering.py  # Feature engineering logic
│   └── requirements.txt    # Python dependencies
│
└── frontend/               # Next.js frontend
    ├── src/
    │   ├── app/           # Next.js app directory
    │   └── components/    # React components
    ├── package.json       # Node.js dependencies
    └── README.md          # Frontend documentation
```

## Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher
- Conda (for managing Python environment)
- npm or yarn (for managing Node.js dependencies)

## Backend Setup

1. Create and activate a conda environment:
```bash
conda create -n movie_analysis python=3.8
conda activate movie_analysis
```

2. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Predict movie rating
  ```json
  {
    "title": "Movie Title",
    "description": "Movie description",
    "tagline": "Movie tagline",
    "release_year": 2024,
    "duration": 120,
    "age_rating": "PG-13",
    "actors": ["Actor 1", "Actor 2"],
    "directors": ["Director 1"],
    "studios": ["Studio 1"]
  }
  ```

## Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Features

- Modern, responsive UI built with Next.js and Tailwind CSS
- Dynamic form for movie details input
- Real-time validation
- Loading states and error handling
- Beautiful results display

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Fill in the movie details:
   - Title
   - Description
   - Tagline
   - Release Year
   - Duration (in minutes)
   - Age Rating
   - Actors (can add multiple)
   - Directors (can add multiple)
   - Studios (can add multiple)
3. Click "Predict Rating" to get the prediction
4. View the predicted rating and confidence level

## Development

### Backend Development

- The backend uses FastAPI for high performance and automatic API documentation
- Feature engineering is handled in `feature_engineering.py`
- The ML model is defined in `model.py`
- API endpoints are defined in `main.py`

### Frontend Development

- Built with Next.js 14 and TypeScript
- Uses Tailwind CSS for styling
- Components are located in `src/components/`
- Main page is in `src/app/page.tsx`

## Troubleshooting

### Backend Issues

1. If the server fails to start:
   - Check if all dependencies are installed
   - Verify Python version (3.8+)
   - Check if the model file exists in the correct location

2. If predictions fail:
   - Check the API logs for error messages
   - Verify the input data format
   - Ensure the model file is loaded correctly

### Frontend Issues

1. If the page doesn't load:
   - Check if the development server is running
   - Verify Node.js version (18+)
   - Check browser console for errors

2. If API calls fail:
   - Ensure the backend server is running
   - Check if CORS is properly configured
   - Verify the API endpoint URL

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 