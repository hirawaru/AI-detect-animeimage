"""
Prepare dataset by splitting into train/val/test sets
"""

import os
import yaml
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config=None, config_path='config.yaml'):
    """
    Split raw data into train/val/test sets
    """
    
    if config is None:
        config = load_config(config_path)
    
    print("=== Preparing Dataset ===")
    
    raw_path = Path(config['data']['raw_path'])
    train_path = Path(config['data']['train_path'])
    val_path = Path(config['data']['val_path'])
    test_path = Path(config['data']['test_path'])
    
    # Create output directories
    for path in [train_path, val_path, test_path]:
        for class_name in ['Natural', 'Synthetic']:
            (path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name in ['Natural', 'Synthetic']:
        class_dir = raw_path / class_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
        
        print(f"\nProcessing {class_name}...")
        
        # Get all images
        images = sorted([f for f in class_dir.glob('*') 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
        
        print(f"  Total images: {len(images)}")
        
        # Split into train/val/test
        train_ratio = config['data']['train_ratio']
        val_ratio = config['data']['val_ratio']
        test_ratio = config['data']['test_ratio']
        
        # First split: train + rest
        train_imgs, rest_imgs = train_test_split(
            images, test_size=(val_ratio + test_ratio), random_state=42
        )
        
        # Second split: val + test
        val_imgs, test_imgs = train_test_split(
            rest_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
        )
        
        print(f"  Train: {len(train_imgs)}")
        print(f"  Val: {len(val_imgs)}")
        print(f"  Test: {len(test_imgs)}")
        
        # Copy files
        for img_path in tqdm(train_imgs, desc=f"  Copying {class_name} to train"):
            shutil.copy2(img_path, train_path / class_name / img_path.name)
        
        for img_path in tqdm(val_imgs, desc=f"  Copying {class_name} to val"):
            shutil.copy2(img_path, val_path / class_name / img_path.name)
        
        for img_path in tqdm(test_imgs, desc=f"  Copying {class_name} to test"):
            shutil.copy2(img_path, test_path / class_name / img_path.name)
    
    print("\n=== Data Preparation Complete ===")
    print(f"Train set: {train_path}")
    print(f"Val set: {val_path}")
    print(f"Test set: {test_path}")


if __name__ == '__main__':
    prepare_data()
