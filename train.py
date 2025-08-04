#!/usr/bin/env python3
"""
Training script for Hybrid Attention U-Net OCT Image Segmentation

This script handles the complete training pipeline including
model initialization, training loop, validation, and evaluation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from models import HybridUNet
from data import load_dataset, preprocess_data, setup_data_loaders
from metrics import (
    dice_coefficient, iou_metric, calculate_metrics, 
    plot_training_metrics, test_preprocessing_visualization,
    visualize_predictions_vs_groundtruth, print_model_summary
)


def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(Config.NUMPY_SEED)
    torch.manual_seed(Config.TORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.TORCH_SEED)


def train_model(model, train_loader, val_loader, epochs=None, learning_rate=None, device=None):
    """
    Train the Hybrid U-Net model using PyTorch.
    
    Args:
        model: HybridUNet model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Training history dictionary
    """
    if epochs is None:
        epochs = Config.EPOCHS
    if learning_rate is None:
        learning_rate = Config.LEARNING_RATE
    if device is None:
        device = Config.DEVICE
        
    print("Starting model training...")
    
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=Config.LR_SCHEDULER_FACTOR, 
        patience=Config.LR_SCHEDULER_PATIENCE
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        train_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coefficient(output, target).item()
            train_iou += iou_metric(output, target).item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                val_dice += dice_coefficient(output, target).item()
                val_iou += iou_metric(output, target).item()
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        avg_train_dice = train_dice / train_batches
        avg_val_dice = val_dice / val_batches
        avg_train_iou = train_iou / train_batches
        avg_val_iou = val_iou / val_batches
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % Config.PRINT_EPOCH_INTERVAL == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            print(f'  Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}')
            print(f'  Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return history


def evaluate_model(model, val_loader, device=None):
    """
    Evaluate the trained model on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = Config.DEVICE
        
    print("\nEvaluating model on test set...")
    model.eval()
    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0
    test_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            test_dice += dice_coefficient(output, target).item()
            test_iou += iou_metric(output, target).item()
            test_batches += 1
            
            # Collect predictions for additional metrics
            pred = torch.softmax(output, dim=1)
            pred = torch.argmax(pred, dim=1)
            
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    
    # Calculate averages
    avg_test_loss = test_loss / test_batches
    avg_test_dice = test_dice / test_batches
    avg_test_iou = test_iou / test_batches
    
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Dice: {avg_test_dice:.4f}")
    print(f"Test IoU: {avg_test_iou:.4f}")
    
    # Calculate additional metrics
    additional_metrics = calculate_metrics(np.array(all_targets), np.array(all_predictions))
    print(f"\nAdditional Metrics:")
    for metric_name, metric_value in additional_metrics.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")
    
    return {
        'test_loss': avg_test_loss,
        'test_dice': avg_test_dice,
        'test_iou': avg_test_iou,
        **additional_metrics
    }


def main():
    """
    Main function to execute the complete pipeline.
    """
    print("=== Hybrid U-Net for OCT Image Segmentation ===")
    
    # Set random seeds
    set_random_seeds()
    
    # Print configuration
    Config.print_config()
    
    # Check for GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Device: {Config.DEVICE}")
    
    # Validate paths
    try:
        Config.validate_paths()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Step 1: Load dataset
    try:
        images, annotations = load_dataset(Config.DATASET_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Step 2: Preprocess data
    try:
        preprocessed_images, preprocessed_masks = preprocess_data(images, annotations)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return
    
    # Step 3: Test preprocessing visualization
    try:
        test_preprocessing_visualization(images, annotations, preprocessed_images, preprocessed_masks)
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    # Step 4: Setup data loaders
    try:
        train_loader, val_loader = setup_data_loaders(preprocessed_images, preprocessed_masks)
    except Exception as e:
        print(f"Error setting up data loaders: {e}")
        return
    
    # Step 5: Initialize model
    model = HybridUNet(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        model_channels=Config.MODEL_CHANNELS,
        use_edge_attention_layers=Config.USE_EDGE_ATTENTION_LAYERS
    )
    
    print_model_summary(model)
    
    # Step 6: Train model
    try:
        history = train_model(model, train_loader, val_loader)
        
        # Step 7: Plot training metrics
        plot_training_metrics(history)
        
        # Step 8: Evaluate model
        evaluation_results = evaluate_model(model, val_loader)
        
        # Step 9: Visualize predictions
        visualize_predictions_vs_groundtruth(model, val_loader, Config.DEVICE)
        
    except Exception as e:
        print(f"Error during training/evaluation: {e}")
        return
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
