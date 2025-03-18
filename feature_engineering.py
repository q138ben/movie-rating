import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def calculate_historical_stats(df, entity_col, rating_col, is_training=True):
    """Calculate historical statistics for entities based on ratings.
    
    Args:
        df: DataFrame containing entity and rating columns
        entity_col: Column name for the entity (actor, director, or studio)
        rating_col: Column name for the rating
        is_training: Whether this is training data (for storing stats)
        
    Returns:
        DataFrame with entity statistics
    """
    # Ensure we have the column
    if entity_col not in df.columns:
        raise KeyError(f"Column {entity_col} not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    stats = df.groupby(entity_col)[rating_col].agg([
        ('avg_rating', 'mean'),
        ('count', 'count')
    ]).reset_index()
    
    # Define "star" entities as those with above average ratings and at least 3 movies
    mean_rating = stats[stats['count'] >= 3]['avg_rating'].mean()
    stats['is_star'] = (stats['avg_rating'] > mean_rating) & (stats['count'] >= 3)
    
    if is_training:
        # Create all necessary directories
        stats_path = Path('data/features/historical_stats')
        stats_path.parent.mkdir(exist_ok=True, parents=True)  # Create data/features
        stats_path.mkdir(exist_ok=True)  # Create historical_stats subdirectory
        
        # Store the statistics for use with validation/test data
        stats.to_csv(stats_path / f'{entity_col}_stats.csv', index=False)
    
    return stats

def get_entity_features(movies_df, entities_df, entity_type, rating_col='rating', is_training=True):
    """Create features for entities (actors, directors, or studios).
    
    Args:
        movies_df: DataFrame containing movie information (indexed by original_id)
        entities_df: DataFrame containing entity relationships
        entity_type: Type of entity ('actor', 'director', or 'studio')
        rating_col: Column name for the rating
        is_training: Whether this is training data
        
    Returns:
        DataFrame with entity-based features
    """
    # Define the correct ID column name based on entity type
    id_column = {
        'actor': 'actor_name',
        'director': 'director_name',
        'studio': 'studio'
    }[entity_type]
    
    # Print debug information
    print(f"\nProcessing {entity_type} features")
    print(f"Available columns in entities_df: {entities_df.columns.tolist()}")
    print(f"Available columns in movies_df: {movies_df.columns.tolist()}")
    
    # Merge movies with entities using original_id
    merged_df = pd.merge(
        entities_df,
        movies_df[[rating_col, 'date']],
        left_on='original_id',
        right_index=True
    )
    
    print(f"Merged DataFrame columns: {merged_df.columns.tolist()}")
    
    if is_training:
        # Calculate historical statistics using only training data
        entity_stats = calculate_historical_stats(
            merged_df,
            id_column,
            rating_col,
            is_training
        )
    else:
        # Load pre-calculated statistics from training data
        stats_path = Path('data/features/historical_stats')
        entity_stats = pd.read_csv(stats_path / f'{id_column}_stats.csv')
    
    # Create features for each movie
    movie_features = {}
    
    # Group by original_id to aggregate entity features
    grouped = merged_df.groupby('original_id')
    
    # Merge with entity statistics
    merged_with_stats = pd.merge(
        merged_df,
        entity_stats,
        on=id_column,
        how='left'
    )
    
    # Calculate features for each movie
    for original_id, group in grouped:
        # Get entity statistics for this movie
        movie_entities = merged_with_stats[merged_with_stats['original_id'] == original_id]
        
        features = {
            f'{entity_type}_count': len(group),
            f'avg_{entity_type}_rating': movie_entities['avg_rating'].mean(),
            f'max_{entity_type}_rating': movie_entities['avg_rating'].max(),
            f'min_{entity_type}_rating': movie_entities['avg_rating'].min(),
            f'star_{entity_type}_count': movie_entities['is_star'].sum(),
            f'has_star_{entity_type}': int(movie_entities['is_star'].any())
        }
        
        movie_features[original_id] = features
    
    return pd.DataFrame.from_dict(movie_features, orient='index')

def load_split_data():
    """Load the train/val/test split data."""
    with open('data/splits/split_indices.json', 'r') as f:
        split_indices = json.load(f)
    
    data_dir = Path('data/filtered')
    movies_df = pd.read_csv(data_dir / 'movies.csv')
    actors_df = pd.read_csv(data_dir / 'actors.csv')
    directors_df = pd.read_csv(data_dir / 'directors.csv')
    studios_df = pd.read_csv(data_dir / 'studios.csv')
    
    # Load embeddings
    poster_embeddings = np.load(data_dir / 'poster_embeddings.npy')
    tagline_embeddings = np.load(data_dir / 'tagline_embeddings.npy')
    description_embeddings = np.load(data_dir / 'description_embeddings.npy')
    
    # Split the data according to indices
    splits = {}
    for split_name in ['train', 'val', 'test']:
        indices = split_indices[split_name]
        splits[split_name] = {
            'movies': movies_df.loc[indices].copy(),
            'embeddings': {
                'poster': poster_embeddings[indices],
                'tagline': tagline_embeddings[indices],
                'description': description_embeddings[indices]
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

def engineer_categorical_numerical_features(split_data):
    """
    Engineer categorical and numerical features from movies and related data.
    
    Args:
        split_data (dict): Dictionary containing train/val/test split data
    
    Returns:
        dict: Dictionary containing engineered features for each split
    """
    # Initialize label encoders
    label_encoders = {
        'theatrical_release_age_rating': LabelEncoder()
    }
    
    # Initialize standard scaler
    scaler = StandardScaler()
    
    # Train encoders and scaler on training data
    train_movies = split_data['train']['movies']
    for col, encoder in label_encoders.items():
        # Fill NaN with 'UNKNOWN' before encoding
        train_movies[col] = train_movies[col].fillna('UNKNOWN')
        encoder.fit(train_movies[col])
    
    # Select and prepare numerical features from movies
    numerical_features = ['minute']
    
    # Prepare features for each split
    engineered_features = {}
    for split_name, split in split_data.items():
        movies_df = split['movies']
        actors_df = split['actors']
        directors_df = split['directors']
        studios_df = split['studios']
        
        # Initialize features DataFrame with original_id as index
        features = pd.DataFrame(index=movies_df['original_id'])
        
        # 1. Basic movie features
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            movies_df[col] = movies_df[col].fillna('UNKNOWN')
            features[f'{col}_encoded'] = encoder.transform(movies_df[col])
        
        # Scale numerical features
        numerical_data = movies_df[numerical_features].fillna(movies_df[numerical_features].mean())
        if split_name == 'train':
            scaled_features = scaler.fit_transform(numerical_data)
        else:
            scaled_features = scaler.transform(numerical_data)
        features['numerical'] = scaled_features
        
        # 2. Temporal features
        features['release_year'] = movies_df['date'].astype(int)
        features['is_recent'] = (features['release_year'] >= 2000).astype(int)
        
        # 3. Actor features
        actor_features = get_entity_features(
            movies_df.set_index('original_id'), 
            actors_df, 
            'actor',
            rating_col='rating', 
            is_training=(split_name == 'train')
        )
        features = features.join(actor_features, how='left')
        
        # 4. Director features
        director_features = get_entity_features(
            movies_df.set_index('original_id'), 
            directors_df, 
            'director',
            rating_col='rating', 
            is_training=(split_name == 'train')
        )
        features = features.join(director_features, how='left')
        
        # 5. Studio features
        studio_features = get_entity_features(
            movies_df.set_index('original_id'), 
            studios_df, 
            'studio',
            rating_col='rating', 
            is_training=(split_name == 'train')
        )
        features = features.join(studio_features, how='left')
        
        # Fill any missing values with 0
        features = features.fillna(0)
        
        # Store features for this split
        engineered_features[split_name] = features
    
    return engineered_features, label_encoders, scaler

def process_text_embeddings(split_data):
    """
    Process text embeddings for taglines and descriptions.
    
    Args:
        split_data (dict): Dictionary containing split data
    
    Returns:
        dict: Processed text embeddings for each split
    """
    text_features = {}
    
    for split in ['train', 'val', 'test']:
        text_features[split] = {
            'tagline': split_data[split]['embeddings']['tagline'],
            'description': split_data[split]['embeddings']['description']
        }
    
    return text_features

def process_image_embeddings(split_data):
    """
    Process image embeddings.
    
    Args:
        split_data (dict): Dictionary containing split data
    
    Returns:
        dict: Processed image embeddings for each split
    """
    image_features = {}
    
    for split in ['train', 'val', 'test']:
        image_features[split] = split_data[split]['embeddings']['poster']
    
    return image_features

def analyze_engineered_features(features_df, targets):
    """
    Analyze and select the most important engineered features.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing engineered features
        targets (np.ndarray): Target values
    
    Returns:
        tuple: (selected_features, feature_names)
    """
    print("Analyzing engineered features...")
    
    # Convert features to numpy array if it's a DataFrame
    if isinstance(features_df, pd.DataFrame):
        feature_names = features_df.columns.tolist()
        features = features_df.values
    else:
        features = features_df
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
    
    # Initialize feature selector
    n_features_to_select = min(10, features.shape[1])
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFE(rf, n_features_to_select=n_features_to_select)
    
    # Fit selector
    selector.fit(features, targets)
    
    # Get selected feature indices
    selected_indices = selector.get_support()
    
    # Get selected feature names
    selected_names = [name for name, selected in zip(feature_names, selected_indices) if selected]
    print(f"Selected features: {', '.join(selected_names)}")
    
    # Get selected features
    selected_features = features[:, selected_indices]
    

    
    # Save plot
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save feature selection results
    results = {
        'selected_features': selected_names,
    }
    
    with open(output_dir / 'feature_selection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return selected_features, selected_names

def reduce_embedding_dimensions(embeddings, embedding_type, variance_threshold=0.95):
    """
    Apply PCA to reduce embedding dimensions while retaining specified variance.
    
    Args:
        embeddings (dict): Dictionary containing embeddings for each split
        embedding_type (str): Type of embedding (tagline, description, or poster)
        variance_threshold (float): Desired explained variance ratio
    
    Returns:
        dict: Reduced embeddings for each split
        PCA: Fitted PCA object
    """
    # Fit PCA on training data
    train_embeddings = embeddings['train'][embedding_type]
    pca = PCA(n_components=min(train_embeddings.shape[1], train_embeddings.shape[0]))
    pca.fit(train_embeddings)
    
    # Find number of components needed
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.axvline(x=n_components, color='g', linestyle='--')
    plt.title(f'Explained Variance Ratio vs. Number of Components ({embedding_type})')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.tight_layout()
    plt.savefig(f'data/features/pca_variance_{embedding_type}.png')
    plt.close()
    
    # Create new PCA with selected number of components
    final_pca = PCA(n_components=n_components)
    reduced_embeddings = {}
    
    # Transform all splits
    for split_name, split_data in embeddings.items():
        reduced_embeddings[split_name] = final_pca.fit_transform(split_data[embedding_type])
    
    return reduced_embeddings, final_pca

def combine_features(engineered_features, text_features, image_features, feature_names):
    """
    Combine all features into a single array.
    
    Args:
        engineered_features (pd.DataFrame): Engineered features
        text_features (dict): Text embeddings
        image_features (np.ndarray): Image embeddings
        feature_names (list): Names of selected engineered features
    
    Returns:
        np.ndarray: Combined features
    """
    # Select only the chosen engineered features
    if isinstance(engineered_features, pd.DataFrame):
        engineered = engineered_features[feature_names].values
    else:
        engineered = engineered_features
    
    # Convert all arrays to float32 and ensure they are 2D
    engineered = np.asarray(engineered, dtype=np.float32)
    if len(engineered.shape) == 1:
        engineered = engineered.reshape(-1, 1)
    
    tagline = np.asarray(text_features['tagline'], dtype=np.float32)
    if len(tagline.shape) == 1:
        tagline = tagline.reshape(-1, 1)
        
    description = np.asarray(text_features['description'], dtype=np.float32)
    if len(description.shape) == 1:
        description = description.reshape(-1, 1)
        
    image = np.asarray(image_features, dtype=np.float32)
    if len(image.shape) == 1:
        image = image.reshape(-1, 1)
    
    # Print shapes for debugging
    print(f"Shapes before concatenation:")
    print(f"- Engineered features: {engineered.shape}")
    print(f"- Tagline embeddings: {tagline.shape}")
    print(f"- Description embeddings: {description.shape}")
    print(f"- Image embeddings: {image.shape}")
    
    # Combine all features
    try:
        combined = np.concatenate([
            engineered,
            tagline,
            description,
            image
        ], axis=1)
        print(f"Combined shape: {combined.shape}")
        return combined.astype(np.float32)  # Ensure final array is float32
    except Exception as e:
        print(f"Error during concatenation: {str(e)}")
        print("Feature types:")
        print(f"- Engineered features: {engineered.dtype}")
        print(f"- Tagline embeddings: {tagline.dtype}")
        print(f"- Description embeddings: {description.dtype}")
        print(f"- Image embeddings: {image.dtype}")
        raise

def get_target_variables(split_data):
    """
    Extract target variables (ratings) from the splits.
    
    Args:
        split_data (dict): Dictionary containing train/val/test split data
    
    Returns:
        dict: Dictionary containing target variables for each split
    """
    targets = {}
    for split_name, split in split_data.items():
        targets[split_name] = split['movies']['rating'].values
    return targets

def engineer_features():
    """
    Main function to orchestrate the feature engineering process.
    This function:
    1. Loads and splits the data
    2. Engineers categorical and numerical features
    3. Processes text embeddings
    4. Processes image embeddings
    5. Analyzes and selects features
    6. Combines all features
    7. Saves the results
    
    Returns:
        dict: Dictionary containing the engineered features and targets for each split
    """
    print("Starting feature engineering process...")
    
    # 1. Load and split data
    print("\n1. Loading and splitting data...")
    split_data = load_split_data()
    
    # 2. Engineer categorical and numerical features
    print("\n2. Engineering categorical and numerical features...")
    engineered_features, label_encoders, scaler = engineer_categorical_numerical_features(split_data)
    
    # Get targets for feature analysis
    targets = get_target_variables(split_data)
    
    # 3. Analyze and select features
    print("\n3. Analyzing engineered features...")
    selected_features, feature_names = analyze_engineered_features(
        engineered_features['train'],
        targets['train']
    )
    
    # 4. Process text embeddings
    print("\n4. Processing text embeddings...")
    text_features = process_text_embeddings(split_data)
    
    # 5. Process image embeddings
    print("\n5. Processing image embeddings...")
    image_features = process_image_embeddings(split_data)
    
    # 6. Combine all features
    print("\n6. Combining all features...")
    combined_features = {}
    feature_info = {}
    
    for split in ['train', 'val', 'test']:
        combined_features[split] = combine_features(
            engineered_features[split],
            text_features[split],
            image_features[split],
            feature_names
        )
    
    # Save feature dimensions for model architecture
    feature_info['dimensions'] = {
        'engineered': len(feature_names),
        'tagline': text_features['train']['tagline'].shape[1],
        'description': text_features['train']['description'].shape[1],
        'image': image_features['train'].shape[1]
    }
    
    # 7. Save results
    print("\n7. Saving engineered features...")
    output_dir = Path('data/features')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save features and targets
    for split in ['train', 'val', 'test']:
        np.save(output_dir / f'{split}_features.npy', combined_features[split])
        np.save(output_dir / f'{split}_targets.npy', targets[split])
    
    # Save feature info
    with open(output_dir / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save preprocessing objects
    preprocessing = {
        'label_encoders': {
            name: {
                'classes': encoder.classes_.tolist()
            }
            for name, encoder in label_encoders.items()
        },
        'scaler': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
    }
    
    with open(output_dir / 'preprocessing.json', 'w') as f:
        json.dump(preprocessing, f, indent=2)
    
    print("\nFeature engineering completed successfully!")
    return combined_features, targets, feature_info

if __name__ == "__main__":
    engineer_features() 