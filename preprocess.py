import torch
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import re

dataset = load_dataset("pranavs28/pokemon_types")
renamed = dataset['train'].rename_column("text", "label") #fixing so pytorch can read it


TYPE_TO_ID = {
    'normal': 0, 'fire': 1, 'water': 2, 'electric': 3, 'grass': 4, 'ice': 5,
    'fighting': 6, 'poison': 7, 'ground': 8, 'flying': 9, 'psychic': 10,
    'bug': 11, 'rock': 12, 'ghost': 13, 'dragon': 14, 'dark': 15,
    'steel': 16, 'fairy': 17
}
ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}

def extract_types(text):
    # Ensure text is a string to avoid AttributeError
    if not isinstance(text, str):
        return []
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    stop_words = {'and', 'type', 'pokemon', 'pokémon'}
    types = [word for word in words if word not in stop_words]
    return [TYPE_TO_ID[t] for t in types if t in TYPE_TO_ID]

def transform_value(example):
    example["label"] = extract_types(example["label"])
    return example

print(renamed.features['label'])  # Schema (Value('string')?)
print(renamed[0]['label'], type(renamed[0]['label']))  # First value
bad_rows = [i for i, ex in enumerate(renamed) if not isinstance(ex['label'], str)]
print("Non-string rows:", bad_rows[:5])  # Indices of issues

transformed = renamed.map(transform_value, batched=False)

split_dataset = transformed.train_test_split(test_size=0.1, seed=42)
split_dataset.save_to_disk("processed_pokemon_dataset")

def display_sample_images(dataset, num_samples=5, split='train'):
    dataset_split = split_dataset[split]
    total_samples = len(dataset_split)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for idx, ax in zip(indices, axes):
        sample = dataset_split[idx]
        image = sample['image']
        pokemon_type_text = sample['label']
        
        if isinstance(pokemon_type_text, list):
             # It's already a list of IDs from transform_value
             print(f"Raw label: {pokemon_type_text}")
             type_display = str(pokemon_type_text)
        else:
             # Fallback if it's still text
             print(f"Raw label: {pokemon_type_text}")
             type_display = str(pokemon_type_text)
        
        ax.imshow(image)
        ax.set_title(f"Type: {type_display}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

display_sample_images(dataset)
