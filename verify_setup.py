import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import (
    FILTERED_DATA_DIR,
    SPLITS_DIR,
    create_directories
)

def verify_python_packages():
    """Verify that all required packages are installed."""
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {pip_name} is installed")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"✗ {pip_name} is missing")
    
    return missing_packages

def verify_data_files():
    """Verify that all required data files exist."""
    required_files = {
        'movies.csv': FILTERED_DATA_DIR / 'movies.csv',
        'actors.csv': FILTERED_DATA_DIR / 'actors.csv',
        'directors.csv': FILTERED_DATA_DIR / 'directors.csv',
        'studios.csv': FILTERED_DATA_DIR / 'studios.csv',
        'poster_embeddings.npy': FILTERED_DATA_DIR / 'poster_embeddings.npy',
        'tagline_embeddings.npy': FILTERED_DATA_DIR / 'tagline_embeddings.npy',
        'description_embeddings.npy': FILTERED_DATA_DIR / 'description_embeddings.npy',
        'split_indices.json': SPLITS_DIR / 'split_indices.json'
    }
    
    missing_files = []
    for name, path in required_files.items():
        if path.exists():
            print(f"✓ Found {name}")
        else:
            missing_files.append(name)
            print(f"✗ Missing {name}")
    
    return missing_files

def verify_cuda():
    """Verify CUDA availability for PyTorch."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA is available ({device_name})")
        return True
    else:
        print("! CUDA is not available, will use CPU")
        return False

def verify_data_consistency():
    """Verify that the data files are consistent."""
    try:
        # Load CSV files
        movies_df = pd.read_csv(FILTERED_DATA_DIR / 'movies.csv')
        actors_df = pd.read_csv(FILTERED_DATA_DIR / 'actors.csv')
        directors_df = pd.read_csv(FILTERED_DATA_DIR / 'directors.csv')
        studios_df = pd.read_csv(FILTERED_DATA_DIR / 'studios.csv')
        
        # Load embeddings
        poster_emb = np.load(FILTERED_DATA_DIR / 'poster_embeddings.npy')
        tagline_emb = np.load(FILTERED_DATA_DIR / 'tagline_embeddings.npy')
        desc_emb = np.load(FILTERED_DATA_DIR / 'description_embeddings.npy')
        
        # Check dimensions
        n_movies = len(movies_df)
        checks = {
            'Poster embeddings': len(poster_emb) == n_movies,
            'Tagline embeddings': len(tagline_emb) == n_movies,
            'Description embeddings': len(desc_emb) == n_movies,
            'Actors references': all(actors_df['original_id'].isin(movies_df['original_id'])),
            'Directors references': all(directors_df['original_id'].isin(movies_df['original_id'])),
            'Studios references': all(studios_df['original_id'].isin(movies_df['original_id']))
        }
        
        inconsistencies = []
        for name, check in checks.items():
            if check:
                print(f"✓ {name} are consistent")
            else:
                inconsistencies.append(name)
                print(f"✗ {name} are inconsistent")
        
        return inconsistencies
    
    except Exception as e:
        print(f"Error during consistency check: {str(e)}")
        return ["Data loading error"]

def main():
    """Run all verifications."""
    print("\nVerifying setup...\n")
    
    # 1. Create directories
    print("1. Creating directory structure...")
    create_directories(verbose=True)
    print()
    
    # 2. Verify Python packages
    print("2. Checking required packages...")
    missing_packages = verify_python_packages()
    print()
    
    # 3. Verify CUDA
    print("3. Checking CUDA availability...")
    has_cuda = verify_cuda()
    print()
    
    # 4. Verify data files
    print("4. Checking required data files...")
    missing_files = verify_data_files()
    print()
    
    # 5. Verify data consistency
    print("5. Checking data consistency...")
    inconsistencies = verify_data_consistency()
    print()
    
    # Summary
    print("\nVerification Summary:")
    print("-" * 50)
    
    if missing_packages:
        print("\n❌ Missing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\n✓ All required packages are installed")
    
    if missing_files:
        print("\n❌ Missing data files:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("\n✓ All required data files are present")
    
    if inconsistencies:
        print("\n❌ Data inconsistencies found:")
        for issue in inconsistencies:
            print(f"  - {issue}")
    else:
        print("\n✓ Data is consistent")
    
    # Final verdict
    if not (missing_packages or missing_files or inconsistencies):
        print("\n✅ All checks passed! You can proceed with running the pipeline.")
        return True
    else:
        print("\n❌ Please fix the issues above before running the pipeline.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 