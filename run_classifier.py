import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_from_disk
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# --- Configuration & Constants ---
TYPE_TO_ID = {
    'normal': 0, 'fire': 1, 'water': 2, 'electric': 3, 'grass': 4, 'ice': 5,
    'fighting': 6, 'poison': 7, 'ground': 8, 'flying': 9, 'psychic': 10,
    'bug': 11, 'rock': 12, 'ghost': 13, 'dragon': 14, 'dark': 15,
    'steel': 16, 'fairy': 17
}
ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}
NUM_CLASSES = 18

# --- Model Definition (Must match training) ---
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # (32, 128, 128)
        self.pool = nn.MaxPool2d(2, 2) # (32, 64, 64)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64x64x64
        self.pool2 = nn.MaxPool2d(2, 2) # (64, 32, 32)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 32x32x128 (block)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # (128, 16, 16)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1) # 16x16x256
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2) # (256, 8, 8)
        
        # Add Adaptive Pooling to reduce dimensions significantly before fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4)) # Output: (256, 4, 4)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Data Preparation ---
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize([0.8412341475486755, 0.8291147947311401, 0.8151265978813171], 
    #                      [0.24166467785835266, 0.24730020761489868, 0.25928303599357605]), 
    transforms.Normalize([0.8412953615188599, 0.8291792869567871, 0.8151875138282776], [0.2233351469039917, 0.2296644002199173, 0.2426573634147644]),
])

def transform_images(batch):
    batch['image'] = [preprocess(image.convert("RGB")) for image in batch['image']]
    
    primary_labels = []
    for labels in batch['label']:
        if len(labels) > 0:
            primary_labels.append(labels[0])
        else:
            primary_labels.append(0)
    
    batch['label'] = primary_labels
    return batch

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    print("Loading dataset...")
    try:
        ds = load_from_disk("processed_pokemon_dataset")
        ds.set_transform(transform_images)
        testloader = DataLoader(ds['test'], batch_size=32, shuffle=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # 2. Load Model
    print("Loading model...")
    net = NeuralNet()
    try:
        net.load_state_dict(torch.load('pokemon_classifier.pth'))
    except FileNotFoundError:
        print("Model file 'pokemon_classifier.pth' not found.")
        exit()
    net.eval()

    # 3. Check Accuracy
    print("Checking accuracy...")
    correct_top1 = 0
    total = 0
    
    # We'll use a threshold for multi-label presence
    threshold = 0.5 

    with torch.no_grad():
        for data in testloader:
            inputs = data['image']
            labels = data['label'] # Long list of class indices

            outputs = net(inputs)
            # outputs shape: [batch_size, NUM_CLASSES]
            
            # Get predicted class index
            _, predicted = torch.max(outputs, 1)
            
            # Compare with ground truth
            correct_top1 += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Top-1 Accuracy on test set: {100 * correct_top1 / total:.2f}%")

    # 4. Display Examples
    print("Displaying examples...")
    
    # Get a fresh batch to show original images (we need to un-normalize or reload raw data)
    # Easiest is to reload a few examples without the transform that standardizes them, 
    # OR just grab them and inverse-normalize, OR load raw from dataset again.
    # Let's use the 'ds' object but grab raw items by temporarily disabling transform? 
    # Actually, ds['test'] items are processed on the fly. 
    # Let's just create a separate list of examples for visualization.
    
    ds_viz = load_from_disk("processed_pokemon_dataset")['test']
    indices = random.sample(range(len(ds_viz)), 5)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    for ax, idx in zip(axes, indices):
        raw_sample = ds_viz[idx] # Raw sample (image is PIL, label is list of IDs)
        
        # Prepare for model
        input_tensor = preprocess(raw_sample['image'].convert("RGB")).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = net(input_tensor)
            # Softmax to get probabilities (optional for visualization, argmax is enough for class)
            probs = F.softmax(output, dim=1)[0]
        
        # Get top prediction
        pred_idx = torch.argmax(probs).item()
            
        pred_label = ID_TO_TYPE[pred_idx]
        
        # Raw label is a list of all types, let's show the primary for consistency
        # or show all true types vs predicted primary
        true_labels = [ID_TO_TYPE[i] for i in raw_sample['label']]
        primary_true = true_labels[0] if true_labels else "Unknown"
        
        ax.imshow(raw_sample['image'])
        ax.set_title(f"True: {primary_true}\nPred: {pred_label}")
        ax.axis('off')
    
    print("Showing plot...")
    plt.tight_layout()
    plt.show()
