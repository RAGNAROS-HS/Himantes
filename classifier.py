import torch
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import re

dataset = load_dataset("pranavs28/pokemon_types")

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.8412341475486755, 0.8291147947311401, 0.8151265978813171], [0.24166467785835266, 0.24730020761489868, 0.25928303599357605]), 
])


def extract_types(text):
    # Capture actual type words, filter stop words, lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    stop_words = {'and', 'type', 'pokemon', 'pokémon'}  
    types = [word for word in words if word not in stop_words]
    return types  

def display_sample_images(dataset, num_samples=5, split='train'):
    dataset_split = dataset[split]
    total_samples = len(dataset_split)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for idx, ax in zip(indices, axes):
        sample = dataset_split[idx]
        image = sample['image']
        pokemon_type_text = sample['text']
        
        # Extract clean types from the text
        types = extract_types(pokemon_type_text)
        print(types)
        type_display = '/'.join(types) if types else 'Unknown'
        
        ax.imshow(image)
        ax.set_title(f"Type: {type_display}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

display_sample_images(dataset)
