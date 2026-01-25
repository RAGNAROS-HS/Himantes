from datasets import load_from_disk
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Define the transform
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.8412341475486755, 0.8291147947311401, 0.8151265978813171], [0.24166467785835266, 0.24730020761489868, 0.25928303599357605]), 
])

def transform_images(batch):
    # Apply the transform to the batch of images
    batch['image'] = [preprocess(image.convert("RGB")) for image in batch['image']]
    return batch

ds = load_from_disk("processed_pokemon_dataset")
ds.set_transform(transform_images)
train_data = ds['train']
test_data = ds['test']

print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

print(train_data[0])

train_data[0].size()