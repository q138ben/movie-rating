from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from predict import load_model

# Import your model and preprocessing code
from model import MultiModalNet, create_model
from feature_engineering import (
    engineer_categorical_numerical_features,
    process_text_embeddings,
    process_image_embeddings,
    combine_features
)

app = FastAPI(title="Movie Rating Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model

with open('../../data/features/feature_info.json', 'r') as f:
    feature_info = json.load(f)

model_path = "models/best_model.pt"

# Load model
model = load_model(
    checkpoint_path=model_path,
    feature_info=feature_info,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)


class MovieInput(BaseModel):
    title: str
    description: str
    tagline: str
    release_year: int
    duration: float
    age_rating: str
    actors: list[str]
    directors: list[str]
    studios: list[str]

@app.get("/")
async def root():
    return {"message": "Movie Rating Prediction API"}

@app.post("/predict")
async def predict_rating(movie: MovieInput):
    try:
        # Convert input to DataFrame format
        movie_data = pd.DataFrame([{
            'title': movie.title,
            'description': movie.description,
            'tagline': movie.tagline,
            'release_year': movie.release_year,
            'duration': movie.duration,
            'theatrical_release_age_rating': movie.age_rating,
            'actor_name': movie.actors,
            'director_name': movie.directors,
            'studio': movie.studios
        }])
        
        # Engineer features
        engineered_features, label_encoders, scaler = engineer_categorical_numerical_features(
            {'train': movie_data},  # Wrap in dict to match expected format
            {'train': np.array([0.0])}  # Dummy target
        )
        
        # Process text embeddings
        text_features = process_text_embeddings(movie_data)
        
        # Process image embeddings (placeholder for now)
        image_features = np.zeros((1, 512))  # Placeholder for image embeddings
        
        # Combine all features
        combined_features = combine_features(
            engineered_features['train'],
            text_features,
            image_features,
            list(engineered_features['train'].columns)  # Feature names
        )
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(combined_features)
        
        # Make prediction
        with torch.no_grad():
            prediction = model({
                'engineered': features_tensor[:, :24],
                'tagline': features_tensor[:, 24:792],
                'description': features_tensor[:, 792:1560],
                'image': features_tensor[:, 1560:]
            })
        
        return {
            "predicted_rating": float(prediction.item()),
            "confidence": "high"  # You can add confidence calculation if needed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 