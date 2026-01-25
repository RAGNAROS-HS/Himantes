import torch
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import re


dataset = load_dataset("pranavs28/pokemon_types")


def calculate_normalization_stats(dataset, split='train'):

    print(f"Calculating normalization statistics from {split} split...")
    
    dataset_split = dataset[split]
    num_samples = len(dataset_split) 
    
    # Transform without normalization
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Initialize accumulators for online mean/std calculation
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    for i in range(num_samples):
        image = dataset_split[i]['image']
        tensor = transform(image)  # Shape: [3, 512, 512]
        
        # Calculate mean and std for each channel
        for c in range(3):
            mean[c] += tensor[c, :, :].mean()
            std[c] += tensor[c, :, :].std()
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_samples} images...")
    
    # Average across all images
    mean = mean / num_samples
    std = std / num_samples
    
    print(f"\nCalculated normalization statistics:")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")
    
    return mean.tolist(), std.tolist()


calculate_normalization_stats(dataset)

# Calculated normalization statistics:
# Mean: [0.8412341475486755, 0.8291147947311401, 0.8151265978813171]
# Std:  [0.24166467785835266, 0.24730020761489868, 0.25928303599357605]
