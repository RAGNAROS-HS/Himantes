# Himantes

Learning Hugging Face + PyTorch by implementing a Pokémon classifier. Applying that to larger Pokémon datasets and subsequently training a diffusion model for new Pokémon creation.

## Project Overview

This project aims to build a comprehensive Pokémon type classification system using PyTorch and Hugging Face's `datasets` library, with future plans to develop a diffusion model for generating new Pokémon designs.

## What Has Been Done So Far

### 1. Dataset Integration
- Integrated the `pranavs28/pokemon_types` dataset from Hugging Face
- Dataset contains Pokémon images with associated type information
- Implemented baseline image preprocessing pipeline with resizing to 512x512

### 2. Normalization Statistics Calculation (`normalization_calculator.py`)
- Created a utility to calculate dataset-specific normalization statistics
- Computes mean and standard deviation across all RGB channels from the training split
- Uses online calculation to process the entire dataset efficiently
- **Calculated values:**
  - Mean: `[0.8412, 0.8291, 0.8151]`
  - Std: `[0.2417, 0.2473, 0.2593]`
- These statistics are used in the preprocessing pipeline for proper data normalization

### 3. Data Visualization (`classifier.py`)
- Implemented `display_sample_images()` function to visualize random samples from the dataset
- Created `extract_types()` utility function to parse and clean Pokémon type information from text
  - Removes stop words like "and", "type", "pokemon"
  - Extracts clean type labels (e.g., "fire/fighting")
- Displays 5 random images with their corresponding types using matplotlib

### 4. Preprocessing Pipeline
- Implemented a torchvision transforms pipeline including:
  - Resize to 512x512 pixels
  - Conversion to tensor format
  - Normalization using calculated dataset statistics

## Project Structure

```
Himantes/
├── classifier.py                  # Main classification script with visualization
├── normalization_calculator.py    # Dataset normalization statistics calculator
├── requirements.txt               # Project dependencies (to be populated)
└── README.md                      # This file
```

## Dependencies

The project uses:
- `torch` - PyTorch deep learning framework
- `datasets` - Hugging Face datasets library
- `torchvision` - Computer vision utilities
- `matplotlib` - Visualization library

## Usage

### Calculate Normalization Statistics
```bash
python normalization_calculator.py
```

### View Sample Images
```bash
python classifier.py
```

## Next Steps

- [ ] Populate `requirements.txt` with dependency versions
- [ ] Implement the actual classifier model architecture
- [ ] Train the classifier on Pokémon type prediction
- [ ] Evaluate model performance
- [ ] Explore diffusion model implementation for Pokémon generation
