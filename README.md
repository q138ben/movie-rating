# Movie Rating Prediction Model

This project implements a multi-modal neural network for predicting movie ratings based on various features including tabular data, text embeddings (taglines and descriptions), and image embeddings (movie posters).

## Project Structure

```
.
├── data/
│   ├── filtered/           # Filtered input data
│   ├── features/          # Engineered features
│   └── splits/            # Train/val/test split indices
├── models/
│   └── checkpoints/       # Model checkpoints
├── plots/                 # Training and evaluation plots
├── predictions/           # Model predictions
├── config.py             # Configuration parameters
├── data_loader.py        # Data loading utilities
├── feature_engineering.py # Feature engineering pipeline
├── model.py              # Model architecture
├── predict.py            # Prediction script
├── train.py              # Training script
├── utils.py              # Utility functions
└── requirements.txt      # Project dependencies
```

## Setup

1. Create a new conda environment:
```bash
conda create -n movie_analysis python=3.8
conda activate movie_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Feature Engineering:
```bash
python feature_engineering.py
```
This script processes raw data and generates engineered features for training.

2. Training:
```bash
python train.py
```
This script trains the multi-modal neural network and saves checkpoints.

3. Prediction:
```bash
python predict.py
```
This script loads a trained model and makes predictions on the test set.

## Model Architecture

The model consists of multiple branches that process different types of features:

1. Tabular Features Branch:
   - Input features → 8 → 4 units
   - Batch normalization and ReLU activation

2. Text Embeddings Branches (Tagline and Description):
   - Input embeddings → 128 → 64 units
   - Batch normalization, ReLU activation, and dropout

3. Image Embeddings Branch:
   - Input embeddings → 128 → 64 units
   - Batch normalization, ReLU activation, and dropout

4. Combined Layers:
   - Concatenated features → 128 → 64 → 32 → 1 unit
   - Batch normalization, ReLU activation, and dropout

## Features

- Multi-modal architecture for processing different types of features
- Feature engineering pipeline for creating relational features
- Early stopping and learning rate scheduling
- Model checkpointing
- Comprehensive evaluation metrics and visualizations
- Modular and maintainable code structure

## Configuration

All configurable parameters are stored in `config.py`. This includes:
- Data and model paths
- Training parameters (batch size, learning rate, etc.)
- Model architecture parameters
- Feature engineering parameters

## Results

Training and evaluation results are saved in:
- `models/checkpoints/`: Model checkpoints
- `plots/`: Training history and prediction analysis plots
- `predictions/`: Test set predictions and metrics

# Movie Rating Case

![Sample poster](sample_posters/0.jpg) ![Sample poster](sample_posters/1.jpg) ![Sample poster](sample_posters/2.jpg)

# Introduction

Executives in the movie industry are curious about the potential of machine learning to predict the success of movies before they are made.

We provide a curated sample of a movie dataset that contains information about roughly 10000 movie titles with features including actors, crew, movie poster, tagline, and more.

Your task is to *predict movie ratings* from different modalities of input features, namely

- Categorical & numerical
- Text
- Image

# Data

## Summary

All data can be found under `data/filtered`. You are free to load it as you wish. The data is provided as

* Tabular data in .csv files:
    * [actors.csv](data/filtered/actors.csv): the *actors* that are credited in the movies
    * [directors.csv](data/filtered/directors.csv): the *directors* are credited in the movies 
    * [movies.csv](data/filtered/movies.csv): movies along with some metadata, including the `rating` variable
    * [studios.csv](data/filtered/studios.csv): the studio(s) that produced the movies
* a [zipped file with posters](data/filtered/posters/posters.zip) - for each movie with a given `movie_id`, there is an image in the archive with the name `{movie_id}.jpg`.
* Pre-computed embeddings for poster, tagline and description data (more information below)

N.b. the column `original_movie_id` exists to preserve a key to the original dataset. You do not need to use it.

## Pre-computed embeddings

If you have limited computational resources at your disposal, processing images and text can be difficult. We therefore provide pre-computed embeddings for the poster, `tagline` and `description` features.

* The image embeddings have been generated using [OpenAI's CLIP model](https://huggingface.co/openai/clip-vit-base-patch32)
* The text embeddings have been generated using Nomic's [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

If you choose to use these embeddings in your model(s), we expect that you possess a basic understanding of how the models that produced them work.

The embeddings provided as numpy arrays with shape `(n_movies, embedding_size)`. They can be loaded into memory using

```
import numpy as np
poster_embeddings = np.load("data/filtered/poster_embeddings.npy")

# To get the embedding for a particular movie:
poster_embeddings[movie_id]
```

# Tasks

**N.b.** you are not expected to spend more than 3-4 hours on this case. We recognize there may not be enough time to cover all tasks in depth. You are free to prioritize the tasks as you deem appropriate and leave some unfinished. Be prepared to reason about any omitted tasks and your rationale for omitting them.

1. Load the dataset.
2. Select the appropriate features to use in your model.
3. Perform preprocessing and cleaning steps of your choice.
4. Model training
    * Train separate models on one or more of the three different modalities (categorical/numerical, text, images) to predict the `rating` property. One modality per model. Choose methods / architectures you find suitable.
    * Validate your models' performance in terms of predictive accuracy and generalizability and compare them. You are free to choose relevant metrics for this task.
5. If you have trained multiple models: combine them or train a new one, to predict `rating` using different modalities of input features. Validate the combined model's performance.
6. Present your code, descriptive analysis, and model performance, for example, in  a Jupyter notebook.

## Optional questions

- Given a movie of your choice, can you modify some aspect of the movie for it to receive a higher rating according to the model(s)?
- Can you, given your model(s), create the ultimate movie? What would it look like?

# Acknowledgements

The dataset provided is based on a [Letterboxd dataset found on Kaggle](https://www.kaggle.com/datasets/gsimonx37/letterboxd/).

# Notes

- We provide the code for processing the full Letterboxd dataset in [this notebook](data/preparation.ipynb). You do not need to run (or even look at) this code to perform any of the tasks.
- When dealing with text and image features, you are free to start from or freeze weights from a pretrained model, provided it has not been trained specifically for this rating prediction task. Be prepared to explain your rationale for doing so.
- This is an open-ended case, and you are encouraged to solve it in a way that you find suitable. Some statements may be vague, such as "compare" the models, and in such cases, you are free to make your own interpretations and assumptions.