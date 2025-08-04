#!/usr/bin/env python3
"""
Metrics and visualization utilities for OCT image segmentation

This module contains functions for calculating performance metrics,
plotting training curves, and visualizing results.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from config import Config


def dice_coefficient(y_pred, y_true, smooth=None):
    """
    Calculate Dice coefficient.
    
    Args:
        y_pred: Predicted tensor
        y_true: Ground truth tensor
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient
    """
    if smooth is None:
        smooth = Config.DICE_SMOOTH
        
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_metric(y_pred, y_true, smooth=None):
    """
    Calculate IoU metric.
    
    Args:
        y_pred: Predicted tensor
        y_true: Ground truth tensor
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    if smooth is None:
        smooth = Config.IOU_SMOOTH
        
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    
    # Calculate Dice coefficient
    intersection = np.sum(y_true_flat == y_pred_flat)
    dice = (2.0 * intersection + 1.0) / (len(y_true_flat) + len(y_pred_flat) + 1.0)
    
    # Calculate IoU
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice': dice,
        'iou': iou
    }


def plot_training_metrics(history, save_path=None):
    """
    Plot training metrics.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    if save_path is None:
        save_path = Config.TRAINING_METRICS_PATH
        
    print("Plotting training metrics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Dice and IoU
    axes[0, 1].plot(epochs, history['train_dice'], 'g-', label='Training Dice')
    axes[0, 1].plot(epochs, history['val_dice'], 'orange', label='Validation Dice')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: IoU
    axes[1, 0].plot(epochs, history['train_iou'], 'm-', label='Training IoU')
    axes[1, 0].plot(epochs, history['val_iou'], 'c-', label='Validation IoU')
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Combined metrics
    axes[1, 1].plot(epochs, history['val_dice'], 'g-', label='Val Dice')
    axes[1, 1].plot(epochs, history['val_iou'], 'orange', label='Val IoU')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training metrics plot saved as {save_path}")


def test_preprocessing_visualization(images, annotations, preprocessed_images, preprocessed_masks, save_path=None):
    """
    Visualize preprocessing results for testing.
    
    Args:
        images: Original images
        annotations: Original annotations
        preprocessed_images: Preprocessed images
        preprocessed_masks: Preprocessed masks
        save_path: Path to save the visualization
    """
    if save_path is None:
        save_path = Config.PREPROCESSING_VIZ_PATH
        
    print("Creating preprocessing test visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image with annotations
    axes[0].imshow(images[0], cmap='gray')
    axes[0].set_title('Original Image with Annotations')
    
    # Plot original annotations
    colors = ['red', 'blue', 'green']
    layer_names = Config.TARGET_LAYERS
    
    for i, (layer_name, color) in enumerate(zip(layer_names, colors)):
        if layer_name in annotations:
            y_coords = annotations[layer_name][0]
            x_coords = range(len(y_coords))
            
            # Filter out NaN values for plotting
            valid_indices = ~np.isnan(y_coords)
            valid_x = [x for x, valid in zip(x_coords, valid_indices) if valid]
            valid_y = y_coords[valid_indices]
            
            if len(valid_x) > 0:
                axes[0].plot(valid_x, valid_y, color=color, label=layer_name, linewidth=2)
    
    axes[0].legend()
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    
    # Preprocessed image with mask overlay
    axes[1].imshow(preprocessed_images[0, :, :, 0], cmap='gray')
    axes[1].imshow(preprocessed_masks[0], alpha=0.3, cmap='jet')
    axes[1].set_title('Preprocessed Image with Mask Overlay')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Preprocessing test visualization saved as {save_path}")


def visualize_predictions_vs_groundtruth(model, data_loader, device, num_samples=4):
    """
    Visualize model predictions vs ground truth.
    
    Args:
        model: Trained model
        data_loader: Data loader for visualization
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    with torch.no_grad():
        for data, target in data_loader:
            if sample_count >= num_samples:
                break
                
            data = data.to(device)
            output = model(data)
            pred = torch.softmax(output, dim=1)
            pred = torch.argmax(pred, dim=1)
            
            # Move to CPU for visualization
            image = data[0, 0].cpu().numpy()
            ground_truth = target[0].cpu().numpy()
            prediction = pred[0].cpu().numpy()
            
            # Plot original image
            axes[sample_count, 0].imshow(image, cmap='gray')
            axes[sample_count, 0].set_title('Original Image')
            axes[sample_count, 0].axis('off')
            
            # Plot ground truth
            axes[sample_count, 1].imshow(ground_truth, cmap='jet')
            axes[sample_count, 1].set_title('Ground Truth')
            axes[sample_count, 1].axis('off')
            
            # Plot prediction
            axes[sample_count, 2].imshow(prediction, cmap='jet')
            axes[sample_count, 2].set_title('Prediction')
            axes[sample_count, 2].axis('off')
            
            sample_count += 1
    
    plt.tight_layout()
    plt.savefig('predictions_vs_groundtruth.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Predictions vs ground truth visualization saved as 'predictions_vs_groundtruth.png'")


def extract_layer_boundaries_from_mask(mask):
    """
    Extract layer boundary coordinates from segmentation mask.
    
    Args:
        mask: Segmentation mask array
        
    Returns:
        Dictionary of layer boundary coordinates
    """
    boundaries = {}
    
    for layer_name, label in Config.LAYER_LABELS.items():
        layer_mask = (mask == label)
        y_coords, x_coords = np.where(layer_mask)
        
        if len(y_coords) > 0:
            # Group by x-coordinate and find the boundary (top-most y for each x)
            boundary_coords = []
            for x in range(mask.shape[1]):
                x_indices = np.where(x_coords == x)[0]
                if len(x_indices) > 0:
                    min_y = np.min(y_coords[x_indices])
                    boundary_coords.append((x, min_y))
                else:
                    boundary_coords.append((x, np.nan))
            
            boundaries[layer_name] = boundary_coords
    
    return boundaries


def print_model_summary(model):
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
    """
    print("=" * 50)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 50)
    print(model)
    print("=" * 50)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50)
