#!/usr/bin/env python3
"""
Data handling, preprocessing, and augmentation for OCT image segmentation

This module contains dataset loading, preprocessing, augmentation,
and the PyTorch Dataset class for OCT images and masks.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import Compose

from config import Config


class OCTDataset(Dataset):
    """Dataset class for OCT images and masks."""
    
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            # Convert to PIL or numpy format for albumentations if needed
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to torch tensors
        if len(image.shape) == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def load_dataset(file_path):
    """
    Load dataset from HDF5 file.
    
    Args:
        file_path: Path to the .h5 file
        
    Returns:
        Tuple of (images, annotations)
    """
    print(f"Loading dataset from {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Load images
        images = np.array(f['images'])
        print(f"Images shape: {images.shape}")
        
        # Load annotations for specific layers
        layers_group = f['layers']
        annotations = {}
        
        for layer in Config.TARGET_LAYERS:
            if layer in layers_group:
                annotations[layer] = np.array(layers_group[layer])
                print(f"{layer} annotations shape: {annotations[layer].shape}")
            else:
                print(f"Warning: {layer} not found in dataset")
    
    return images, annotations


def create_augmentation_pipeline():
    """
    Create augmentation pipeline using Albumentations.
    
    Returns:
        Albumentations Compose object
    """
    return Compose([
        A.VerticalFlip(p=Config.VERTICAL_FLIP_PROB),
        A.HorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
        A.CLAHE(p=Config.CLAHE_PROB),
        A.GaussianBlur(blur_limit=Config.GAUSSIAN_BLUR_LIMIT, p=Config.GAUSSIAN_BLUR_PROB),
        A.InvertImg(p=Config.INVERT_IMG_PROB),
        A.Equalize(p=Config.EQUALIZE_PROB)
    ])


def preprocess_data(images, annotations, target_size=None):
    """
    Preprocess images and annotations.
    
    Args:
        images: Input images array
        annotations: Dictionary of annotations
        target_size: Target image size (height, width)
        
    Returns:
        Tuple of (preprocessed_images, preprocessed_masks)
    """
    if target_size is None:
        target_size = Config.TARGET_SIZE
        
    print("Preprocessing data...")
    
    batch_size, original_height, original_width = images.shape
    target_height, target_width = target_size
    
    # Initialize preprocessed arrays
    preprocessed_images = np.zeros((batch_size, target_height, target_width, 1))
    preprocessed_masks = np.zeros((batch_size, target_height, target_width))
    
    # Create augmentation pipeline
    augment = create_augmentation_pipeline()
    
    # Scaling factor for annotations
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    
    for i in range(batch_size):
        # Convert to uint8 and normalize to 0-255 range
        img = images[i].astype(np.float32)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Convert grayscale to RGB for albumentations compatibility
        img_rgb = np.stack([img, img, img], axis=-1)
        
        # Resize image using albumentations
        augmented = A.Resize(target_height, target_width, interpolation=1)(image=img_rgb)
        img_resized = augmented['image']
        
        # Apply augmentations
        if np.random.random() > Config.AUGMENTATION_PROBABILITY:
            augmented = augment(image=img_resized)
            img_resized = augmented['image']
        
        # Convert back to grayscale and normalize to 0-1
        img_grayscale = np.mean(img_resized, axis=2)
        img_grayscale = img_grayscale.astype(np.float32) / 255.0
        preprocessed_images[i, :, :, 0] = img_grayscale
        
        # Create combined mask for three layers
        mask = np.zeros((target_height, target_width))
        
        # Process each layer annotation
        for layer_name, label in Config.LAYER_LABELS.items():
            if layer_name in annotations:
                layer_coords = annotations[layer_name][i]
                
                # Scale and create mask
                for x in range(len(layer_coords)):
                    y_coord_raw = layer_coords[x]
                    
                    # Skip NaN values
                    if np.isnan(y_coord_raw):
                        continue
                        
                    y_coord = int(y_coord_raw * height_scale)
                    x_coord = int(x * width_scale)
                    
                    if 0 <= y_coord < target_height and 0 <= x_coord < target_width:
                        mask[y_coord, x_coord] = label
        
        preprocessed_masks[i] = mask
    
    print(f"Preprocessed images shape: {preprocessed_images.shape}")
    print(f"Preprocessed masks shape: {preprocessed_masks.shape}")
    
    return preprocessed_images, preprocessed_masks


def create_segmentation_mask(annotations, image_shape, layer_labels=None):
    """
    Create segmentation mask from layer annotations.
    
    Args:
        annotations: Dictionary of layer annotations
        image_shape: Target image shape (height, width)
        layer_labels: Dictionary mapping layer names to labels
        
    Returns:
        Segmentation mask array
    """
    if layer_labels is None:
        layer_labels = Config.LAYER_LABELS
        
    height, width = image_shape
    mask = np.zeros((height, width))
    
    for layer_name, label in layer_labels.items():
        if layer_name in annotations:
            layer_coords = annotations[layer_name]
            
            for x in range(len(layer_coords)):
                y_coord = layer_coords[x]
                
                # Skip NaN values
                if np.isnan(y_coord):
                    continue
                    
                y_coord = int(y_coord)
                
                if 0 <= y_coord < height and 0 <= x < width:
                    mask[y_coord, x] = label
    
    return mask


def setup_data_loaders(images, masks, batch_size=None, test_size=None, random_state=None):
    """
    Setup train and validation data loaders.
    
    Args:
        images: Preprocessed images
        masks: Preprocessed masks
        batch_size: Batch size for data loaders
        test_size: Fraction of data to use for testing
        random_state: Random state for train/test split
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if test_size is None:
        test_size = Config.TEST_SIZE
    if random_state is None:
        random_state = Config.RANDOM_STATE
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, masks,
        test_size=test_size,
        random_state=random_state,
        stratify=None
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create datasets
    train_dataset = OCTDataset(X_train.squeeze(-1), y_train)  # Remove channel dimension
    val_dataset = OCTDataset(X_test.squeeze(-1), y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
