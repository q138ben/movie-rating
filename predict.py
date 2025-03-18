import torch
import numpy as np
from pathlib import Path
import json

from data_loader import load_features
from model import create_model

def load_model(checkpoint_path, feature_info, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        feature_info (dict): Feature dimension information
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
    # Create model with same architecture
    model = create_model(feature_info['dimensions'])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model

def predict(model, features, feature_info, device):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        features (np.ndarray): Feature matrix
        feature_info (dict): Feature dimension information
        device: Device to run predictions on
    
    Returns:
        np.ndarray: Predicted ratings
    """
    # Split features into modalities
    start_idx = 0
    feature_dict = {}
    
    for feature_type, dim in feature_info['dimensions'].items():
        end_idx = start_idx + dim
        feature_dict[feature_type] = torch.FloatTensor(
            features[:, start_idx:end_idx]
        ).to(device)
        start_idx = end_idx
    
    # Make predictions
    with torch.no_grad():
        predictions = model(feature_dict).cpu().numpy()
    
    return predictions

def evaluate_model(predictions, targets):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): True ratings
    
    Returns:
        dict: Dictionary of metrics
    """
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load features and targets
    features, targets, feature_info = load_features()
    
    # Load model
    model = load_model(
        checkpoint_path='models/checkpoints/best_model.pt',
        feature_info=feature_info,
        device=device
    )
    
    # Make predictions on test set
    test_predictions = predict(model, features['test'], feature_info, device)
    
    # Evaluate predictions
    metrics = evaluate_model(test_predictions, targets['test'])
    
    print("\nTest Set Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    # Save predictions and metrics
    output_dir = Path('predictions')
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'test_predictions.npy', test_predictions)
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main() 