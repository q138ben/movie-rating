import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

class MultiModalNet(nn.Module):
    """Neural network for multi-modal movie rating prediction."""
    def __init__(self, feature_dims):
        super(MultiModalNet, self).__init__()
        
        # Dimensions for each modality
        self.tabular_dim = feature_dims['tabular']
        self.tagline_dim = feature_dims['tagline']
        self.description_dim = feature_dims['description']
        self.poster_dim = feature_dims['poster']
        
        # Tabular features branch (reduced size due to few features)
        self.tabular_branch = nn.Sequential(
            nn.Linear(self.tabular_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        
        # Text embeddings branch (tagline)
        self.tagline_branch = nn.Sequential(
            nn.Linear(self.tagline_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Text embeddings branch (description)
        self.description_branch = nn.Sequential(
            nn.Linear(self.description_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Image embeddings branch
        self.poster_branch = nn.Sequential(
            nn.Linear(self.poster_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined layers (4 + 64 + 64 + 64 = 196)
        combined_dim = 4 + 64 + 64 + 64
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x_tabular, x_tagline, x_description, x_poster):
        # Process each modality
        tabular_features = self.tabular_branch(x_tabular)
        tagline_features = self.tagline_branch(x_tagline)
        description_features = self.description_branch(x_description)
        poster_features = self.poster_branch(x_poster)
        
        # Concatenate features
        combined = torch.cat([
            tabular_features,
            tagline_features,
            description_features,
            poster_features
        ], dim=1)
        
        # Final prediction
        return self.combined_layers(combined).squeeze()

class MovieDataset(Dataset):
    """Custom Dataset for multi-modal movie rating prediction."""
    def __init__(self, features, targets, feature_info):
        # Split features into modalities
        self.tabular_features = torch.FloatTensor(features[:, :feature_info['dimensions']['engineered']])
        start_idx = feature_info['dimensions']['engineered']
        
        self.tagline_features = torch.FloatTensor(
            features[:, start_idx:start_idx + feature_info['dimensions']['tagline']]
        )
        start_idx += feature_info['dimensions']['tagline']
        
        self.description_features = torch.FloatTensor(
            features[:, start_idx:start_idx + feature_info['dimensions']['description']]
        )
        start_idx += feature_info['dimensions']['description']
        
        self.poster_features = torch.FloatTensor(
            features[:, start_idx:start_idx + feature_info['dimensions']['image']]
        )
        
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'tabular': self.tabular_features[idx],
            'tagline': self.tagline_features[idx],
            'description': self.description_features[idx],
            'poster': self.poster_features[idx],
            'target': self.targets[idx]
        }

def load_data():
    """Load preprocessed features and targets."""
    data_dir = Path('data/features')
    
    # Load features and targets
    features = {}
    targets = {}
    for split in ['train', 'val']:
        features[split] = np.load(data_dir / f'{split}_features.npy')
        targets[split] = np.load(data_dir / f'{split}_targets.npy')
    
    # Load feature info
    with open(data_dir / 'feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    return features, targets, feature_info

def train_neural_network(train_features, train_targets, val_features, val_targets, feature_info, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train neural network model."""
    # Create data loaders
    train_dataset = MovieDataset(train_features, train_targets, feature_info)
    val_dataset = MovieDataset(val_features, val_targets, feature_info)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    feature_dims = {
        'tabular': feature_info['dimensions']['engineered'],
        'tagline': feature_info['dimensions']['tagline'],
        'description': feature_info['dimensions']['description'],
        'poster': feature_info['dimensions']['image']
    }
    
    model = MultiModalNet(feature_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    n_epochs = 300
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 50
    counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Move all inputs to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                inputs['tabular'],
                inputs['tagline'],
                inputs['description'],
                inputs['poster']
            )
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Move all inputs to device
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
                targets = batch['target'].to(device)
                
                outputs = model(
                    inputs['tabular'],
                    inputs['tagline'],
                    inputs['description'],
                    inputs['poster']
                )
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'data/models/best_nn_model.pt')
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('data/models/nn_training_curves.png')
    plt.close()
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('data/models/best_nn_model.pt'))
    return model

def evaluate_model(model, features, targets, split_name, feature_info, model_type='nn', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate model performance on a specific split."""
    if model_type == 'nn':
        model.eval()
        dataset = MovieDataset(features, targets, feature_info)
        loader = DataLoader(dataset, batch_size=32)
        
        all_predictions = []
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
                outputs = model(
                    inputs['tabular'],
                    inputs['tagline'],
                    inputs['description'],
                    inputs['poster']
                )
                all_predictions.append(outputs.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
    else:
        predictions = model.predict(features)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    results = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(targets, predictions),
        'r2': r2_score(targets, predictions)
    }
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 10], [0, 10], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(f'{model_type.upper()} - {split_name.capitalize()} Set Predictions')
    plt.savefig(f'data/models/{model_type}_{split_name}_predictions.png')
    plt.close()
    
    return results

def train_and_evaluate_models(features, targets, feature_info):
    """Train models and evaluate their performance."""
    results = {}
    
    # Neural Network
    print("\nTraining Neural Network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_model = train_neural_network(
        features['train'], targets['train'],
        features['val'], targets['val'],
        feature_info,
        device
    )
    
    # Evaluate on train and validation sets
    results['Neural Network'] = {
        'train': evaluate_model(nn_model, features['train'], targets['train'], 'train', feature_info, model_type='nn', device=device),
        'val': evaluate_model(nn_model, features['val'], targets['val'], 'val', feature_info, model_type='nn', device=device)
    }
    
    return results, {'nn': nn_model}

if __name__ == "__main__":
    print("Loading data...")
    features, targets, feature_info = load_data()
    
    # Create models directory
    Path('data/models').mkdir(exist_ok=True)
    
    # Train and evaluate models
    results, models = train_and_evaluate_models(features, targets, feature_info)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame()
    for model_name, model_results in results.items():
        for split in ['train', 'val']:
            for metric, value in model_results[split].items():
                comparison.loc[f"{model_name} - {split}", metric] = value
    
    print("\nModel Comparison (Training and Validation):")
    print(comparison.round(4))
    
    # Save results
    comparison.to_csv('data/models/model_comparison.csv')
    
    print("\nResults and visualizations saved in data/models/")
    
    # After selecting the best model based on validation performance,
    # we can evaluate it on the val set
    print("\nEvaluating best model on val set...")
    
    # Find best model based on validation RMSE
    best_model_name = min(results.keys(), 
                         key=lambda k: results[k]['val']['rmse'])
    
    print(f"\nBest model: {best_model_name}")
    
    # Evaluate best model on val set
    if best_model_name == 'Neural Network':
        val_results = evaluate_model(
            models['nn'], 
            features['val'], 
            targets['val'], 
            'val', 
            feature_info,
            model_type='nn',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    else:
        model_type = 'rf' if best_model_name == 'Random Forest' else 'xgb'
        val_results = evaluate_model(
            models[model_type],
            features['val'],
            targets['val'],
            'val',
            feature_info,
            model_type=model_type
        )
    
    print("\nval Set Performance:")
    for metric, value in val_results.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Save val results
    val_results_df = pd.DataFrame([val_results], index=[f"{best_model_name} - val"])
    val_results_df.to_csv('data/models/val_results.csv') 