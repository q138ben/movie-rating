import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalNet(nn.Module):
    def __init__(self, feature_dims):
        """
        Initialize the multi-modal neural network.
        
        Args:
            feature_dims (dict): Dictionary containing dimensions for each feature type
                {
                    'engineered': int,
                    'tagline': int,
                    'description': int,
                    'image': int
                }
        """
        super(MultiModalNet, self).__init__()
        
        # Tabular features branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(feature_dims['engineered'], 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        
        # Text embeddings branches
        self.tagline_branch = nn.Sequential(
            nn.Linear(feature_dims['tagline'], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.description_branch = nn.Sequential(
            nn.Linear(feature_dims['description'], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Image embeddings branch
        self.image_branch = nn.Sequential(
            nn.Linear(feature_dims['image'], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Combined layers
        combined_dim = 4 + 64 + 64 + 64  # 196
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (dict): Dictionary containing input features
                {
                    'engineered': tensor,
                    'tagline': tensor,
                    'description': tensor,
                    'image': tensor
                }
        
        Returns:
            tensor: Predicted rating
        """
        # Process each modality
        tabular_out = self.tabular_branch(x['engineered'])
        tagline_out = self.tagline_branch(x['tagline'])
        description_out = self.description_branch(x['description'])
        image_out = self.image_branch(x['image'])
        
        # Concatenate all features
        combined = torch.cat([
            tabular_out,
            tagline_out,
            description_out,
            image_out
        ], dim=1)
        
        # Final prediction
        return self.combined_layers(combined).squeeze(-1)

def create_model(feature_dims):
    """
    Create and initialize the model.
    
    Args:
        feature_dims (dict): Dictionary containing dimensions for each feature type
    
    Returns:
        MultiModalNet: Initialized model
    """
    model = MultiModalNet(feature_dims)
    
    # Initialize weights using Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    return model 