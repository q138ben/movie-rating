import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from data_loader import load_features
from model import create_model

class MovieDataset(Dataset):
    def __init__(self, features, targets, feature_info):
        """
        Dataset class for movie ratings prediction.
        
        Args:
            features (np.ndarray): Feature matrix
            targets (np.ndarray): Target ratings
            feature_info (dict): Feature dimension information
        """
        self.targets = torch.FloatTensor(targets)
        
        # Split features into modalities
        start_idx = 0
        self.features = {}
        
        for feature_type, dim in feature_info['dimensions'].items():
            end_idx = start_idx + dim
            self.features[feature_type] = torch.FloatTensor(features[:, start_idx:end_idx])
            start_idx = end_idx
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.features.items()}, self.targets[idx]

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_features, batch_targets in train_loader:
        # Move data to device
        batch_features = {k: v.to(device) for k, v in batch_features.items()}
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            # Move data to device
            batch_features = {k: v.to(device) for k, v in batch_features.items()}
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(mse))
    
    return {
        'loss': float(total_loss / len(val_loader)),
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    checkpoint_dir,
    patience=5
):
    """
    Train the model with early stopping.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        patience: Number of epochs to wait for improvement before early stopping
    
    Returns:
        dict: Training history
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    best_val_loss = float('inf')
    best_epoch = 0
    no_improvement = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = float(train_epoch(model, train_loader, criterion, optimizer, device))
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint if validation loss improved
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            no_improvement = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': float(val_metrics['loss']),
                'val_rmse': float(val_metrics['rmse']),
                'val_mae': float(val_metrics['mae'])
            }, checkpoint_dir / 'best_model.pt')
            
            # Save training history
            with open(checkpoint_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
        else:
            no_improvement += 1
        
        # Early stopping
        if no_improvement >= patience:
            print(f"\nEarly stopping triggered. Best epoch was {best_epoch+1}")
            break
    
    return history

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    features, targets, feature_info = load_features()
    
    # Create datasets
    train_dataset = MovieDataset(features['train'], targets['train'], feature_info)
    val_dataset = MovieDataset(features['val'], targets['val'], feature_info)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(feature_info['dimensions'])
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device,
        checkpoint_dir='models/checkpoints',
        patience=5
    )
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 