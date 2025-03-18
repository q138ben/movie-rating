from pathlib import Path

# Data paths
DATA_DIR = Path('data')
FILTERED_DATA_DIR = DATA_DIR / 'filtered'
FEATURES_DIR = DATA_DIR / 'features'
HISTORICAL_STATS_DIR = FEATURES_DIR / 'historical_stats'
SPLITS_DIR = DATA_DIR / 'splits'
SPLITS_PATH = SPLITS_DIR / 'split_indices.json'

# Model paths
MODEL_DIR = Path('models')
CHECKPOINTS_DIR = MODEL_DIR / 'checkpoints'
MODEL_SUMMARY_PATH = MODEL_DIR / 'model_summary.txt'

# Output paths
PLOTS_DIR = Path('plots')
PREDICTIONS_DIR = Path('predictions')

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5

# Model architecture parameters
TABULAR_HIDDEN_DIMS = [8, 4]
TEXT_HIDDEN_DIMS = [128, 64]
IMAGE_HIDDEN_DIMS = [128, 64]
COMBINED_HIDDEN_DIMS = [128, 64, 32]
DROPOUT_RATE = 0.3

# Feature engineering parameters
PCA_VARIANCE_THRESHOLD = 0.95
MIN_MOVIES_FOR_STAR = 3

# Random seed for reproducibility
RANDOM_SEED = 42

def create_directories(verbose=True):
    """
    Create all required directories if they don't exist.
    
    Args:
        verbose (bool): Whether to print status messages
    """
    directories = [
        (DATA_DIR, "Main data directory"),
        (FILTERED_DATA_DIR, "Filtered data directory"),
        (FEATURES_DIR, "Features directory"),
        (HISTORICAL_STATS_DIR, "Historical statistics directory"),
        (SPLITS_DIR, "Data splits directory"),
        (MODEL_DIR, "Model directory"),
        (CHECKPOINTS_DIR, "Model checkpoints directory"),
        (PLOTS_DIR, "Plots directory"),
        (PREDICTIONS_DIR, "Predictions directory")
    ]
    
    for directory, description in directories:
        if not directory.exists():
            directory.mkdir(parents=True)
            if verbose:
                print(f"Created {description} at: {directory}")
        else:
            if verbose:
                print(f"Found existing {description} at: {directory}")

if __name__ == "__main__":
    print("Creating directory structure...")
    create_directories(verbose=True)
    print("\nDirectory structure setup completed!") 