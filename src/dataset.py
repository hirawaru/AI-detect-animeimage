"""
Dataset loader for AI Image Detection
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AIImageDataset(Dataset):
    """Custom Dataset for AI-generated vs Natural images"""
    
    def __init__(self, root_dir, transform=None, class_names=['Natural', 'Synthetic']):
        """
        Args:
            root_dir (str): Root directory containing image class folders
            transform (callable, optional): Optional transform to be applied on images
            class_names (list): List of class folder names
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        self.images = []
        self.labels = []
        
        # Load image paths and labels, validating images and skipping corrupted files
        skipped = 0
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    try:
                        # Validate image file quickly by opening and verifying header
                        with Image.open(img_path) as im:
                            im.verify()
                        self.images.append(img_path)
                        self.labels.append(class_idx)
                    except Exception as e:
                        skipped += 1
                        print(f"Skipped corrupt image {img_path}: {type(e).__name__}: {e}")
        
        print(f"Loaded {len(self.images)} images from {root_dir} (skipped {skipped} corrupt files)")
        for class_idx, class_name in enumerate(class_names):
            count = sum(1 for label in self.labels if label == class_idx)
            print(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {type(e).__name__}: {e}")
            # Return a dummy image in case of error to keep DataLoader stable
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=384, mode='train', normalize_mean=None, normalize_std=None):
    """
    Get image transforms for preprocessing
    
    Args:
        image_size (int): Target image size
        mode (str): 'train' for augmentation, 'val'/'test' for no augmentation
        normalize_mean (list): Normalization mean values
        normalize_std (list): Normalization std values
    
    Returns:
        transforms.Compose: Composed transforms
    """
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
