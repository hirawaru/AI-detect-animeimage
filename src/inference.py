"""
Inference script for AI Image Detection
"""

import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.dataset import get_transforms
from src.model import AIImageClassifier


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def predict_image(image_path, model, transform, device, class_names=['Natural', 'Synthetic']):
    """
    Predict label for a single image
    
    Args:
        image_path (str): Path to image file
        model (nn.Module): Trained model
        transform (callable): Image transform
        device (torch.device): Device to use
        class_names (list): List of class names
    
    Returns:
        dict: Prediction results
    """
    model.eval()
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    probabilities = {class_names[i]: probs[0, i].item() for i in range(len(class_names))}
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'probabilities': probabilities,
        'image_path': str(image_path)
    }


def predict_batch(image_dir, model, transform, device, class_names=['Natural', 'Synthetic']):
    """
    Predict labels for all images in a directory
    
    Args:
        image_dir (str): Path to directory containing images
        model (nn.Module): Trained model
        transform (callable): Image transform
        device (torch.device): Device to use
        class_names (list): List of class names
    
    Returns:
        list: List of prediction results
    """
    image_dir = Path(image_dir)
    results = []
    
    for image_path in sorted(image_dir.glob('*')):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            result = predict_image(image_path, model, transform, device, class_names)
            if result:
                results.append(result)
                print(f"{image_path.name}: {result['predicted_class']} ({result['confidence']:.4f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Inference for AI Image Detection')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory with images')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    
    # Load model
    model = AIImageClassifier(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout=config['model']['dropout']
    )
    
    model_path = args.model_path or f"{config['output']['checkpoint_path']}/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Loaded model from {model_path}")
    
    # Get transform
    transform = get_transforms(
        image_size=config['preprocessing']['image_size'],
        mode='val',
        normalize_mean=config['preprocessing']['normalize_mean'],
        normalize_std=config['preprocessing']['normalize_std']
    )
    
    # Predict
    if args.image_path:
        result = predict_image(args.image_path, model, transform, device)
        if result:
            print("\n=== Prediction Result ===")
            print(f"Image: {result['image_path']}")
            print(f"Predicted: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: {result['probabilities']}")
    
    elif args.image_dir:
        results = predict_batch(args.image_dir, model, transform, device)
        print(f"\n=== Batch Prediction Complete ===")
        print(f"Processed {len(results)} images")
    
    else:
        print("Please provide either --image_path or --image_dir")


if __name__ == '__main__':
    main()
