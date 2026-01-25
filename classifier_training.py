from datasets import load_from_disk
from torch.utils.data import DataLoader
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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
   # transforms.Normalize([0.8412341475486755, 0.8291147947311401, 0.8151265978813171], [0.24166467785835266, 0.24730020761489868, 0.25928303599357605]), 
    transforms.Normalize([0.8412953615188599, 0.8291792869567871, 0.8151875138282776], [0.2233351469039917, 0.2296644002199173, 0.2426573634147644]),
])

def transform_images(batch):
    # Apply the transform to the batch of images
    batch['image'] = [preprocess(image.convert("RGB")) for image in batch['image']]
    
    # Convert labels to single integer (primary type)
    # batch['label'] contains a list of lists. We take the first element of each inner list.
    primary_labels = []
    for labels in batch['label']:
        # Taking the first label as the primary type
        if len(labels) > 0:
            primary_labels.append(labels[0])
        else:
            # Fallback for empty label lists (shouldn't happen in this dataset but good for safety)
            primary_labels.append(0) 
            
    batch['label'] = primary_labels
    return batch

ds = load_from_disk("processed_pokemon_dataset")

# Create class weights for loss balancing
# Calculate weights BEFORE setting the transform to access raw integer labels
print("Calculating class weights...")
raw_train_data = ds['train']
num_classes = 18
class_counts = torch.zeros(num_classes)
for labels in raw_train_data['label']:
    # Only count the primary type
    if len(labels) > 0:
        primary = labels[0]
        if 0 <= primary < num_classes:
            class_counts[primary] += 1

# pos_weight is for BCE, for CrossEntropyLoss we use just 'weight'
# Formula: total / (num_classes * count) or similar inverse frequency
total_samples = len(raw_train_data)
# Simple inverse frequency
weights = total_samples / (num_classes * class_counts)
# Handle potential division by zero if a class is missing (add epsilon or clamp)
weights = torch.where(class_counts > 0, weights, torch.ones_like(weights))

print(f"Class weights: {weights}")

ds.set_transform(transform_images)
train_data = ds['train']
test_data = ds['test']

print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
print("Image shape:", train_data[0]['image'].shape)



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
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Reduced from 262k inputs to 4k inputs
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 18)

    def forward(self, x):  # Dims: 128->64->32->16->8
            x = self.pool(F.relu(self.conv1(x)))  # 64x64x32
            x = self.pool(F.relu(self.conv2(x)))  # 32x32x64
            # Better: group
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool2(x)  # 16x16x128
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.pool3(x)  # 8x8x256
            x = self.avgpool(x) # 4x4x256
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            return x


net = NeuralNet()

loss_function = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


trainloader = DataLoader(train_data, batch_size=4, shuffle=True)
testloader = DataLoader(test_data, batch_size=4, shuffle=False)

for epoch in range(10):
    print(f"Training epoch {epoch}")

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs = data['image']
        labels = data['label']

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0



torch.save(net.state_dict(), 'pokemon_classifier.pth')  

#pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129
#this might help with GPU issues