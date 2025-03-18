import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data(data_dir='data/filtered'):
    """
    Load raw data from CSV files and embeddings.
    
    Args:
        data_dir (str): Directory containing the data files
    
    Returns:
        tuple: movies_df, actors_df, directors_df, studios_df, embeddings
    """
    data_dir = Path(data_dir)
    
    # Load CSV files
    movies_df = pd.read_csv(data_dir / 'movies.csv')
    actors_df = pd.read_csv(data_dir / 'actors.csv')
    directors_df = pd.read_csv(data_dir / 'directors.csv')
    studios_df = pd.read_csv(data_dir / 'studios.csv')
    
    # Load embeddings
    embeddings = {
        'poster': np.load(data_dir / 'poster_embeddings.npy'),
        'tagline': np.load(data_dir / 'tagline_embeddings.npy'),
        'description': np.load(data_dir / 'description_embeddings.npy')
    }
    
    return movies_df, actors_df, directors_df, studios_df, embeddings

def load_split_data(split_path='data/splits/split_indices.json', data_dir='data/filtered'):
    """
    Load data split into train/val/test sets.
    
    Args:
        split_path (str): Path to split indices JSON file
        data_dir (str): Directory containing the data files
    
    Returns:
        dict: Dictionary containing train/val/test splits
    """
    # Load split indices
    with open(split_path, 'r') as f:
        split_indices = json.load(f)
    
    # Load all data
    movies_df, actors_df, directors_df, studios_df, embeddings = load_data(data_dir)
    
    # Split the data according to indices
    splits = {}
    for split_name in ['train', 'val', 'test']:
        indices = split_indices[split_name]
        splits[split_name] = {
            'movies': movies_df.loc[indices].copy(),
            'embeddings': {
                'poster': embeddings['poster'][indices],
                'tagline': embeddings['tagline'][indices],
                'description': embeddings['description'][indices]
            }
        }
        # Get original_ids for the split
        original_ids = set(splits[split_name]['movies']['original_id'])
        splits[split_name].update({
            'actors': actors_df[actors_df['original_id'].isin(original_ids)].copy(),
            'directors': directors_df[directors_df['original_id'].isin(original_ids)].copy(),
            'studios': studios_df[studios_df['original_id'].isin(original_ids)].copy()
        })
    
    return splits

def load_features(features_dir='data/features'):
    """
    Load preprocessed features and targets.
    
    Args:
        features_dir (str): Directory containing feature files
    
    Returns:
        tuple: features, targets, feature_info
    """
    features_dir = Path(features_dir)
    
    # Load features and targets
    features = {}
    targets = {}
    for split in ['train', 'val', 'test']:
        features[split] = np.load(features_dir / f'{split}_features.npy', allow_pickle=True)
        targets[split] = np.load(features_dir / f'{split}_targets.npy', allow_pickle=True)
    
    # Load feature info
    with open(features_dir / 'feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    return features, targets, feature_info

def load_preprocessing_objects(features_dir='data/features'):
    """
    Load preprocessing objects (label encoders, scaler).
    
    Args:
        features_dir (str): Directory containing preprocessing objects
    
    Returns:
        dict: Dictionary containing preprocessing objects
    """
    with open(Path(features_dir) / 'preprocessing.json', 'r') as f:
        preprocessing = json.load(f)
    
    return preprocessing 