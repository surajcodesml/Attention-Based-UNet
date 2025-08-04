# Hybrid Attention U-Net for OCT Image Segmentation

This project implements a Hybrid Attention Mechanism-Based U-Net for segmenting three specific sub-retinal layers (ILM, PR1, BM) in Optical Coherence Tomography (OCT) images.

## Project Structure

The codebase has been modularized for better organization and maintainability:

```
├── config.py          # Configuration and hyperparameters
├── models.py          # Model architecture components
├── data.py           # Data loading, preprocessing, and augmentation
├── metrics.py        # Metrics calculation and visualization
├── train.py          # Main training script
├── UNet_hybrid.py    # Legacy file (redirects to new structure)
└── README.md         # This file
```

## File Descriptions

### `config.py`
Contains all configuration settings, hyperparameters, and paths:
- Dataset configuration (paths, target layers, labels)
- Image preprocessing parameters
- Training hyperparameters (batch size, epochs, learning rate)
- Model architecture settings
- Augmentation parameters
- Output paths

### `models.py`
Contains all neural network components:
- `EdgeAttentionBlock`: Attention mechanism for early layers focusing on edge information
- `SpatialAttentionBlock`: Attention mechanism for deeper layers focusing on spatial features
- `EncoderBlock`: Encoder block with optional attention mechanism
- `DecoderBlock`: Decoder block with skip connections
- `HybridUNet`: Complete U-Net model with hybrid attention mechanisms

### `data.py`
Handles all data-related operations:
- `OCTDataset`: PyTorch Dataset class for OCT images and masks
- `load_dataset()`: Load dataset from HDF5 files
- `preprocess_data()`: Image preprocessing and mask creation
- `create_augmentation_pipeline()`: Data augmentation setup
- `setup_data_loaders()`: Create train/validation data loaders

### `metrics.py`
Contains evaluation metrics and visualization functions:
- `dice_coefficient()`: Calculate Dice coefficient
- `iou_metric()`: Calculate IoU metric
- `calculate_metrics()`: Comprehensive performance metrics
- `plot_training_metrics()`: Plot training curves
- `test_preprocessing_visualization()`: Visualize preprocessing results
- `visualize_predictions_vs_groundtruth()`: Compare predictions with ground truth

### `train.py`
Main training script with complete pipeline:
- Model initialization
- Training loop with validation
- Early stopping and learning rate scheduling
- Model evaluation
- Result visualization

## Usage

### Quick Start
```bash
python train.py
```

### Running the Legacy File
```bash
python UNet_hybrid.py
```
This will automatically redirect to the new modular structure.

### Customizing Configuration
Edit `config.py` to modify:
- Dataset path
- Hyperparameters
- Model architecture
- Training settings

Example:
```python
# config.py
class Config:
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0005
    TARGET_SIZE = (256, 128)
```

## Model Architecture

The Hybrid U-Net uses two types of attention mechanisms:

1. **Edge Attention Block**: Applied to early encoder layers (1-2)
   - Focuses on edge information using Canny-like operations
   - Helps preserve boundary details

2. **Spatial Attention Block**: Applied to deeper layers (3-5)
   - Uses global max and average pooling
   - Focuses on spatial feature relationships

## Training Features

- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Comprehensive Metrics**: Dice, IoU, Precision, Recall, F1-score
- **Visualization**: Training curves and prediction comparisons
- **Reproducibility**: Fixed random seeds

## Requirements

- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- albumentations
- h5py

## Dataset Format

The dataset should be in HDF5 format with:
- `images`: OCT images array
- `layers/ILM`: ILM layer annotations
- `layers/PR1`: PR1 layer annotations  
- `layers/BM`: BM layer annotations

## Configuration Options

Key configuration parameters in `config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `EPOCHS` | Number of training epochs | 30 |
| `BATCH_SIZE` | Training batch size | 4 |
| `LEARNING_RATE` | Initial learning rate | 0.001 |
| `TARGET_SIZE` | Image resize dimensions | (512, 256) |
| `NUM_CLASSES` | Number of segmentation classes | 4 |
| `EARLY_STOPPING_PATIENCE` | Early stopping patience | 10 |

## Output Files

The training process generates:
- `best_model_weights.pth`: Best model weights
- `training_metrics.png`: Training curves visualization
- `preprocessing_test.png`: Preprocessing visualization
- `predictions_vs_groundtruth.png`: Prediction comparisons

## Model Performance

The model is evaluated using multiple metrics:
- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index
- **Precision/Recall/F1**: Classification metrics
- **Loss**: Cross-entropy loss

## Tips for Best Results

1. **Data Quality**: Ensure high-quality annotations
2. **Preprocessing**: Proper image normalization is crucial
3. **Augmentation**: Use appropriate augmentations for medical images
4. **Hyperparameters**: Tune learning rate and batch size for your dataset
5. **GPU**: Use CUDA-enabled GPU for faster training

## Contributing

The modular structure makes it easy to:
- Add new attention mechanisms in `models.py`
- Implement new metrics in `metrics.py`
- Modify preprocessing in `data.py`
- Adjust hyperparameters in `config.py`

## Legacy Support

The original `UNet_hybrid.py` file is preserved for backward compatibility but redirects to the new modular structure. This ensures existing workflows continue to work while encouraging adoption of the improved organization.