"""
Evaluation script for AI Image Detection
"""

import os
import json
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.dataset import AIImageDataset, get_transforms
from src.model import AIImageClassifier


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, test_loader, device, class_names=['Natural', 'Synthetic']):
    """
    Evaluate model on test set and compute metrics
    
    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_score': f1_score(all_labels, all_preds),
        'roc_auc': roc_auc_score(all_labels, all_probs[:, 1]),
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
    metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
    
    return metrics, all_preds, all_labels, all_probs, cm


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_pr_curve(precision, recall, save_path=None):
    """Plot and save Precision-Recall curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    plt.close()


def evaluate(config=None, config_path='config.yaml', model_path=None, results_output_path=None):
    """Main evaluation function"""
    
    if config is None:
        config = load_config(config_path)
    
    if results_output_path is None:
        results_output_path = config['output']['results_path']
    
    os.makedirs(results_output_path, exist_ok=True)

    print("=== AI Image Detection - Evaluation ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = AIImageClassifier(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout=config['model']['dropout']
    )
    
    if model_path is None:
        model_path = f"{config['output']['checkpoint_path']}/best_model.pth"
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    print(f"Loaded model from {model_path}")
    
    # Load test data
    test_transform = get_transforms(
        image_size=config['preprocessing']['image_size'],
        mode='val',
        normalize_mean=config['preprocessing']['normalize_mean'],
        normalize_std=config['preprocessing']['normalize_std']
    )
    
    test_dataset = AIImageDataset(config['data']['test_path'], transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Evaluate
    metrics, preds, labels, probs, cm = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n=== Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\n" + classification_report(labels, preds, target_names=['Natural', 'Synthetic']))
    
    # Save metrics
    results_json_path = f"{results_output_path}/metrics.json"
    with open(results_json_path, 'w') as f:
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc']),
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report']
        }, f, indent=2)
    print(f"\nMetrics saved to {results_json_path}")
    
    # Plot results
    plot_confusion_matrix(cm, ['Natural', 'Synthetic'], 
                         f"{results_output_path}/confusion_matrix.png")
    plot_roc_curve(metrics['roc_curve']['fpr'], metrics['roc_curve']['tpr'], 
                   metrics['roc_auc'], f"{results_output_path}/roc_curve.png")
    plot_pr_curve(metrics['pr_curve']['precision'], metrics['pr_curve']['recall'],
                  f"{results_output_path}/pr_curve.png")
    
    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AI Image Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--results_path', type=str, default=None, help='Path to save results')
    args = parser.parse_args()
    
    evaluate(config_path=args.config, model_path=args.model_path, results_output_path=args.results_path)
