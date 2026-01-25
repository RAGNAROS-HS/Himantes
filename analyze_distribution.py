from datasets import load_from_disk
import torch
import collections

try:
    ds = load_from_disk("processed_pokemon_dataset")
    train_data = ds['train']
    
    # TYPE_TO_ID map from preprocess.py
    TYPE_TO_ID = {
        'normal': 0, 'fire': 1, 'water': 2, 'electric': 3, 'grass': 4, 'ice': 5,
        'fighting': 6, 'poison': 7, 'ground': 8, 'flying': 9, 'psychic': 10,
        'bug': 11, 'rock': 12, 'ghost': 13, 'dragon': 14, 'dark': 15,
        'steel': 16, 'fairy': 17
    }
    ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}
    
    counts = collections.defaultdict(int)
    total_images = len(train_data)
    
    print(f"Total training images: {total_images}")
    
    for i in range(total_images):
        labels = train_data[i]['label']
        # labels is a list of type IDs
        if len(labels) > 0:
             primary_label = labels[0]
             counts[primary_label] += 1
             
    print("\nClass Distribution:")
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    for label_id, count in sorted_counts:
        type_name = ID_TO_TYPE.get(label_id, "Unknown")
        percentage = (count / total_images) * 100
        print(f"{type_name:<10} (ID: {label_id}): {count} images ({percentage:.2f}%)")
        
except Exception as e:
    print(f"Error: {e}")
