# Himantes

**STATUS: ABANDONED**

> **Post-Mortem:** This project has been discontinued. While a functional classification pipeline was established, the model struggled to generalize effectively. The primary blockers were:
> 1.  **High-Variance Anatomy:** Pokémon designs exhibit extreme morphological diversity within the same type (e.g., *Charmander* vs. *Torkoal*), making visual patterns difficult for a standard CNN to capture without massive data.
> 2.  **Insufficient Data:** The available dataset provided too few examples per class to overcome the high variance, likely leading to overfitting or poor feature representation.

## Project Overview

This project aimed to build a deep learning system to classify Pokémon types from images using PyTorch and Hugging Face's `datasets`. The goal was to train a custom CNN to predict up to two types for a given Pokémon image.

## Implemented Features

### 1. Data Pipeline & Analysis
- **Integration**: Integrated the `pranavs28/pokemon_types` dataset.
- **Distribution Analysis**: Mapped class imbalances using `analyze_distribution.py` to understand the prevalence of types like Water/Normal vs. rarer types like Ice/Ghost.
- **Normalization**: Calculated custom normalization statistics (Mean/Std) specifically for this dataset using `normalization_calculator.py` to ensure optimal model convergence.
  - Mean: `[0.8412, 0.8291, 0.8151]`
  - Std: `[0.2417, 0.2473, 0.2593]`

### 2. Model Architecture
- **Custom CNN**: Implemented a 6-layer Convolutional Neural Network with:
  - Adaptive Average Pooling `(4,4)` to handle spatial feature variations.
  - Dropout (`0.5`) in fully connected layers to mitigate overfitting.
- **Multi-Label Support**: Designed for 18 distinct types using a multi-hot encoding scheme.

### 3. Training Strategy (`classifier_training.py`)
- **Loss Function**: Implemented `BCEWithLogitsLoss` with **calculated class weights** (`pos_weights`) to counter the significant dataset imbalance (e.g., Water types being much more common than Ice).
- **Optimizer**: Utilized SGD with momentum (`0.9`) for stable convergence.
- **Artifacts**: Saves the best model state to `pokemon_classifier.pth`.

### 4. Inference & Visualization (`run_classifier.py`)
- **Preprocessing**: Robust pipeline resizing images to 512x512 and applying the calculated normalization.
- **Inference Engine**: Loads the trained model and applies a threshold-based prediction logic.
- **Visualizer**: Uses `extract_types()` and Matplotlib to display side-by-side comparisons of Ground Truth vs. Predicted types.

## Project Structure

```
Himantes/
├── classifier_training.py     # Main training script (Model definition + Training Loop)
├── run_classifier.py          # Inference script for visualization and testing
├── analyze_distribution.py    # Utility to analyze dataset class distribution
├── normalization_calculator.py # Script to compute dataset mean/std
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Dependencies
- `torch` (PyTorch)
- `datasets` (Hugging Face)
- `torchvision`
- `matplotlib`
- `numpy`

## Usage (Historical)

### 1. Train the Model
```bash
python classifier_training.py
```

### 2. Run Inference (Visualize Results)
```bash
python run_classifier.py
```

### 3. Utilities
```bash
python normalization_calculator.py
python analyze_distribution.py
```
