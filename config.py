#!/usr/bin/env python3
"""
Configuration file for Hybrid Attention U-Net OCT Image Segmentation

This file contains all hyperparameters, paths, and configuration settings
for the OCT image segmentation project.
"""

import os
import torch

class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    # Dataset Configuration
    DATASET_PATH = "/home/suraj/Git/SCR-Progression/Nemours_Jing_RL_Annotated.h5"
    TARGET_LAYERS = ['ILM', 'PR1', 'BM']
    LAYER_LABELS = {'ILM': 1, 'PR1': 2, 'BM': 3}
    
    # Image Preprocessing
    TARGET_SIZE = (512, 256)  # (height, width)
    INPUT_CHANNELS = 1
    NUM_CLASSES = 4  # Background + 3 layers
    
    # Training Configuration
    BATCH_SIZE = 4
    EPOCHS = 30
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Model Configuration
    MODEL_CHANNELS = [64, 128, 256, 512, 1024]
    USE_EDGE_ATTENTION_LAYERS = [True, True, False, False, False]  # For encoder layers 1-5
    
    # Training Settings
    EARLY_STOPPING_PATIENCE = 10
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.1
    
    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Augmentation Settings
    AUGMENTATION_PROBABILITY = 0.5
    VERTICAL_FLIP_PROB = 0.5
    HORIZONTAL_FLIP_PROB = 0.5
    CLAHE_PROB = 1.0
    GAUSSIAN_BLUR_PROB = 0.5
    GAUSSIAN_BLUR_LIMIT = (1, 3)
    INVERT_IMG_PROB = 0.1
    EQUALIZE_PROB = 0.1
    
    # Metric Settings
    DICE_SMOOTH = 1e-6
    IOU_SMOOTH = 1e-6
    
    # Output Settings
    MODEL_SAVE_PATH = "best_model_weights.pth"
    PREPROCESSING_VIZ_PATH = "preprocessing_test.png"
    TRAINING_METRICS_PATH = "training_metrics.png"
    
    # Progress Reporting
    PRINT_EPOCH_INTERVAL = 10
    
    # Random Seeds
    NUMPY_SEED = 42
    TORCH_SEED = 42
    
    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist."""
        if not os.path.exists(cls.DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {cls.DATASET_PATH}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration settings."""
        print("=== Configuration Settings ===")
        print(f"Dataset Path: {cls.DATASET_PATH}")
        print(f"Target Size: {cls.TARGET_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Model Channels: {cls.MODEL_CHANNELS}")
        print(f"Target Layers: {cls.TARGET_LAYERS}")
        print("=" * 30)
