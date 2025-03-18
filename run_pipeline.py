import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
from datetime import datetime

# Import our modules
from verify_setup import main as verify_setup
from config import create_directories
from feature_engineering import engineer_features
from train import main as train_main
from predict import main as predict_main
from utils import plot_training_history, analyze_predictions

def setup_environment():
    """Setup and verify the environment."""
    print("\n1. Setting up and verifying environment...")
    if not verify_setup():
        print("\nEnvironment verification failed. Please fix the issues above before proceeding.")
        sys.exit(1)
    print("\nEnvironment verification successful!")

def run_feature_engineering():
    """Run the feature engineering pipeline."""
    print("\n2. Running feature engineering...")
    try:
        engineer_features()
        print("Feature engineering completed successfully!")
    except Exception as e:
        print(f"Error during feature engineering: {str(e)}")
        sys.exit(1)

def train_model():
    """Train the model."""
    print("\n3. Training model...")
    try:
        train_main()
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        sys.exit(1)

def make_predictions():
    """Make predictions using the trained model."""
    print("\n4. Making predictions...")
    try:
        predict_main()
        print("Predictions completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

def analyze_results():
    """Analyze and visualize results."""
    print("\n5. Analyzing results...")
    try:
        # Plot training history
        history_path = Path('models/checkpoints/training_history.json')
        if history_path.exists():
            plot_training_history(history_path)
            print("Training history plots created successfully!")
        else:
            print("Warning: No training history found to plot")
        
        # Analyze predictions
        predictions_path = Path('predictions/test_predictions.npy')
        if predictions_path.exists():
            predictions = np.load(predictions_path)
            targets = np.load(Path('data/features/test_targets.npy'))
            analyze_predictions(predictions, targets)
            print("Prediction analysis completed successfully!")
        else:
            print("Warning: No predictions found to analyze")
        
    except Exception as e:
        print(f"Error during results analysis: {str(e)}")
        sys.exit(1)

def main():
    """Run the complete pipeline."""
    # Print header
    print("\n" + "="*60)
    print("Movie Rating Prediction Pipeline".center(60))
    print("="*60 + "\n")
    
    start_time = datetime.now()
    print(f"Starting pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run pipeline steps
    setup_environment()
    run_feature_engineering()
    train_model()
    make_predictions()
    analyze_results()
    
    # Print completion summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("Pipeline Completion Summary".center(60))
    print("="*60)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {duration}")
    print("\nAll steps completed successfully!")
    print("\nOutputs can be found in:")
    print("  - Feature files:     data/features/")
    print("  - Model checkpoints: models/checkpoints/")
    print("  - Analysis plots:    plots/")
    print("  - Predictions:       predictions/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {str(e)}")
        sys.exit(1) 