import pandas as pd
import numpy as np
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load all necessary datasets."""
    data_dir = Path('data/filtered')
    
    # Load CSV data
    movies_df = pd.read_csv(data_dir / 'movies.csv')
    actors_df = pd.read_csv(data_dir / 'actors.csv')
    directors_df = pd.read_csv(data_dir / 'directors.csv')
    studios_df = pd.read_csv(data_dir / 'studios.csv')
    
    # Load embeddings
    poster_embeddings = np.load(data_dir / 'poster_embeddings.npy')
    tagline_embeddings = np.load(data_dir / 'tagline_embeddings.npy')
    description_embeddings = np.load(data_dir / 'description_embeddings.npy')
    
    return {
        'movies': movies_df,
        'actors': actors_df,
        'directors': directors_df,
        'studios': studios_df,
        'embeddings': {
            'poster': poster_embeddings,
            'tagline': tagline_embeddings,
            'description': description_embeddings
        }
    }

def time_based_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split the data based on time, with validation and test sets randomly split from the same time period.
    
    Args:
        data (dict): Dictionary containing all datasets
        train_ratio (float): Desired proportion for training set (default: 0.7)
        val_ratio (float): Desired proportion for validation set (default: 0.15)
        test_ratio (float): Desired proportion for test set (default: 0.15)
    
    Returns:
        dict: Dictionary containing train, validation, and test indices and data
    """
    movies_df = data['movies']
    
    # Sort movies by date and get cumulative counts per year
    year_counts = movies_df['date'].value_counts().sort_index()
    total_movies = len(movies_df)
    cumulative_ratio = year_counts.cumsum() / total_movies
    
    # Find split year based on train ratio
    train_end_year = cumulative_ratio[cumulative_ratio <= train_ratio].index.max()
    
    # Create train mask
    train_mask = movies_df['date'] <= train_end_year
    
    # Get indices for recent period (after train_end_year)
    recent_indices = movies_df[~train_mask].index.tolist()
    
    # Randomly shuffle recent indices
    np.random.shuffle(recent_indices)
    
    # Calculate sizes for validation and test sets
    n_recent = len(recent_indices)
    n_val = int(n_recent * (val_ratio / (val_ratio + test_ratio)))
    
    # Split recent indices into validation and test
    val_indices = recent_indices[:n_val]
    test_indices = recent_indices[n_val:]
    
    # Get indices for each split
    split_indices = {
        'train': movies_df[train_mask].index.tolist(),
        'val': val_indices,
        'test': test_indices
    }
    
    # Create split datasets
    split_data = {}
    
    # Split movies dataframe
    split_data['movies'] = {
        'train': movies_df.loc[split_indices['train']],
        'val': movies_df.loc[split_indices['val']],
        'test': movies_df.loc[split_indices['test']]
    }
    
    # Get original_ids for each split
    split_original_ids = {
        'train': set(split_data['movies']['train']['original_id'].tolist()),
        'val': set(split_data['movies']['val']['original_id'].tolist()),
        'test': set(split_data['movies']['test']['original_id'].tolist())
    }
    
    # Split embeddings
    split_data['embeddings'] = {}
    for embed_type, embeddings in data['embeddings'].items():
        split_data['embeddings'][embed_type] = {
            'train': embeddings[split_indices['train']],
            'val': embeddings[split_indices['val']],
            'test': embeddings[split_indices['test']]
        }
    
    # Split related dataframes (actors, directors, studios) using original_id
    for name in ['actors', 'directors', 'studios']:
        df = data[name]
        split_data[name] = {
            'train': df[df['original_id'].isin(split_original_ids['train'])],
            'val': df[df['original_id'].isin(split_original_ids['val'])],
            'test': df[df['original_id'].isin(split_original_ids['test'])]
        }
    
    # Calculate and store split statistics
    stats = {
        'years': {
            'train': sorted(split_data['movies']['train']['date'].unique().tolist()),
            'val': sorted(split_data['movies']['val']['date'].unique().tolist()),
            'test': sorted(split_data['movies']['test']['date'].unique().tolist())
        },
        'counts': {
            'movies': {split: len(df) for split, df in split_data['movies'].items()},
            'actors': {split: len(df) for split, df in split_data['actors'].items()},
            'directors': {split: len(df) for split, df in split_data['directors'].items()},
            'studios': {split: len(df) for split, df in split_data['studios'].items()}
        }
    }
    
    # Calculate actual ratios
    total_movies = sum(stats['counts']['movies'].values())
    stats['ratios'] = {
        split: count/total_movies 
        for split, count in stats['counts']['movies'].items()
    }
    
    # Save split information
    output_dir = Path('data/splits')
    output_dir.mkdir(exist_ok=True)
    
    # Save indices and statistics
    with open(output_dir / 'split_indices.json', 'w') as f:
        json.dump(split_indices, f)
    
    with open(output_dir / 'split_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return split_data, split_indices, stats

if __name__ == "__main__":
    # Load all data
    print("Loading data...")
    data = load_data()
    
    # Perform time-based split
    print("\nPerforming time-based split...")
    split_data, split_indices, stats = time_based_split(data)
    
    # Print split statistics
    print("\nSplit Statistics:")
    print("Years in each split:")
    for split, years in stats['years'].items():
        print(f"{split.capitalize()}: {min(years)} - {max(years)}")
    
    print("\nNumber of items in each split:")
    for data_type, counts in stats['counts'].items():
        print(f"\n{data_type.capitalize()}:")
        for split, count in counts.items():
            print(f"  {split}: {count}")
    
    print("\nActual split ratios:")
    for split, ratio in stats['ratios'].items():
        print(f"{split}: {ratio:.1%}")
    
    print("\nTime periods:")
    for split in ['train', 'val', 'test']:
        years = stats['years'][split]
        print(f"{split}: {min(years)} - {max(years)}") 