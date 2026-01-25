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
    
    # Convert labels to multi-hot vectors for multi-label classification
    num_classes = 18
    # batch['label'] contains a list of lists, where each inner list has the type IDs
    multi_hot_labels = []
    for labels in batch['label']:
        # Create a zero tensor of shape (num_classes,)
        target = torch.zeros(num_classes)
        for label_id in labels:
             if 0 <= label_id < num_classes:
                target[label_id] = 1.0
        multi_hot_labels.append(target)
    
    batch['label'] = multi_hot_labels
    return batch

ds = load_from_disk("processed_pokemon_dataset")
ds.set_transform(transform_images)
train_data = ds['train']
test_data = ds['test']

print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
print("Image shape:", train_data[0]['image'].shape)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # (32, 512, 512)
        self.pool = nn.MaxPool2d(2, 2) # (32, 256, 256)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 256x256x64
        self.pool2 = nn.MaxPool2d(2, 2) # (64, 128, 128)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 256x256x128 (block)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # (128, 64, 64)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1) # 128x128x256
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2) # (256, 32, 32)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)  # 64x64 →32x32 wait no: track below
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 18)

    def forward(self, x):  # Dims: 512→256→128→64→32→16 flatten
            x = self.pool(F.relu(self.conv1(x)))  # 256x256x32
            x = self.pool(F.relu(self.conv2(x)))  # 128x128x64? Wait pool after conv2 single
            # Better: group
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool2(x)  # 64x64x128
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.pool3(x)  # 32x32x256 → flatten 256*32*32=262k
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            # remove log_softmax for BCEWithLogitsLoss which expects raw logits
            return x


net = NeuralNet()
dict_keys = ds['train'].features
print(dict_keys)
# We need to ensure the labels are floats for BCEWithLogitsLoss
loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
