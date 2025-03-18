import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def plot_training_history(history_path, output_dir='plots'):
    """
    Plot training history metrics.
    
    Args:
        history_path (str): Path to training history JSON file
        output_dir (str): Directory to save plots
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot([x['loss'] for x in history['val_metrics']], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'loss_history.png')
    plt.close()
    
    # Plot validation metrics
    plt.figure(figsize=(10, 6))
    plt.plot([x['rmse'] for x in history['val_metrics']], label='RMSE')
    plt.plot([x['mae'] for x in history['val_metrics']], label='MAE')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'validation_metrics.png')
    plt.close()

def analyze_predictions(predictions, targets, output_dir='plots'):
    """
    Analyze model predictions.
    
    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): True ratings
        output_dir (str): Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot prediction vs actual
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.title('Predicted vs Actual Ratings')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.grid(True)
    plt.savefig(output_dir / 'predictions_vs_actual.png')
    plt.close()
    
    # Plot error distribution
    errors = predictions - targets
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(output_dir / 'error_distribution.png')
    plt.close()
    
    # Calculate error statistics
    error_stats = {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'median_error': float(np.median(errors)),
        'max_error': float(np.max(np.abs(errors)))
    }
    
    with open(output_dir / 'error_statistics.json', 'w') as f:
        json.dump(error_stats, f, indent=2)

def save_model_summary(model, feature_info, output_path='models/model_summary.txt'):
    """
    Save model architecture summary.
    
    Args:
        model: PyTorch model
        feature_info (dict): Feature dimension information
        output_path (str): Path to save summary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        # Write feature dimensions
        f.write("Feature Dimensions:\n")
        f.write("-" * 50 + "\n")
        for feature_type, dim in feature_info['dimensions'].items():
            f.write(f"{feature_type}: {dim}\n")
        f.write("\n")
        
        # Write model architecture
        f.write("Model Architecture:\n")
        f.write("-" * 50 + "\n")
        f.write(str(model))
        
        # Write total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write("\n\nModel Summary:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n") 