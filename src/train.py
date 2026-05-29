"""
Training script for AI Image Detection
"""

import os
import json
import yaml
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import AIImageDataset, get_transforms
from src.model import AIImageClassifier


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create necessary directories"""
    os.makedirs(config['output']['model_save_path'], exist_ok=True)
    os.makedirs(config['output']['checkpoint_path'], exist_ok=True)
    os.makedirs(config['output']['results_path'], exist_ok=True)
    os.makedirs(f"{config['output']['results_path']}/logs", exist_ok=True)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train(config=None, config_path='config.yaml', resume=None):
    """Main training function"""
    
    if config is None:
        config = load_config(config_path)
    
    # Coerce types for common numeric config values to avoid YAML parsing issues
    try:
        config['training']['batch_size'] = int(config['training']['batch_size'])
    except Exception:
        config['training']['batch_size'] = int(float(config['training']['batch_size']))

    config['training']['num_epochs'] = int(config['training']['num_epochs'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])

    # Ensure num_workers and image size are integers
    config['num_workers'] = int(config.get('num_workers', 0))
    config['preprocessing']['image_size'] = int(config['preprocessing']['image_size'])

    # Normalize boolean-like strings for pretrained flag
    if isinstance(config['model'].get('pretrained'), str):
        config['model']['pretrained'] = config['model']['pretrained'].lower() in ('true', '1', 'yes')
    
    print("=== AI Image Detection - Training ===")
    print(f"Config: {config_path}")
    
    # Setup
    setup_directories(config)
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = AIImageClassifier(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    print(f"Model: {config['model']['name']}")
    
    # Load data
    train_transform = get_transforms(
        image_size=config['preprocessing']['image_size'],
        mode='train',
        normalize_mean=config['preprocessing']['normalize_mean'],
        normalize_std=config['preprocessing']['normalize_std']
    )
    val_transform = get_transforms(
        image_size=config['preprocessing']['image_size'],
        mode='val',
        normalize_mean=config['preprocessing']['normalize_mean'],
        normalize_std=config['preprocessing']['normalize_std']
    )
    
    train_dataset = AIImageDataset(config['data']['train_path'], transform=train_transform)
    val_dataset = AIImageDataset(config['data']['val_path'], transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], 
                          weight_decay=config['training']['weight_decay'])
    
    if config['training']['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'])
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # TensorBoard
    writer = SummaryWriter(f"{config['output']['results_path']}/logs")

    # Load existing training history if present (useful when resuming)
    history_path = f"{config['output']['results_path']}/training_history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except Exception:
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    else:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Determine starting epoch (supports resume via checkpoint or existing history)
    start_epoch = len(history.get('train_loss', []))

    # Training bookkeeping
    best_val_loss = float('inf')
    patience_counter = 0

    # If resume checkpoint provided, attempt to load model/optimizer/scheduler state
    if resume:
        if os.path.exists(resume):
            try:
                ckpt = torch.load(resume, map_location=device)
                # If full checkpoint dict with keys
                if isinstance(ckpt, dict):
                    if 'model_state_dict' in ckpt:
                        model.load_state_dict(ckpt['model_state_dict'])
                    else:
                        try:
                            model.load_state_dict(ckpt)
                        except Exception:
                            pass
                    if 'optimizer_state_dict' in ckpt:
                        try:
                            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                        except Exception:
                            pass
                    if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
                        try:
                            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                        except Exception:
                            pass
                    if 'best_val_loss' in ckpt:
                        best_val_loss = ckpt.get('best_val_loss', best_val_loss)
                    # If checkpoint contains epoch, resume from next
                    if 'epoch' in ckpt:
                        start_epoch = ckpt.get('epoch', 0) + 1
                else:
                    # assume it's a state_dict
                    try:
                        model.load_state_dict(ckpt)
                    except Exception:
                        pass
                print(f"Resuming training from checkpoint: {resume}, start_epoch={start_epoch}")
            except Exception as e:
                print(f"Warning: failed to load resume checkpoint {resume}: {e}")
        else:
            print(f"Warning: resume checkpoint {resume} not found")
    

    for epoch in range(start_epoch, config['training']['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config['training']['num_epochs']
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model (save full checkpoint including optimizer/scheduler state)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = f"{config['output']['checkpoint_path']}/best_model.pth"
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
                    'best_val_loss': best_val_loss,
                }, checkpoint_path)
            except Exception:
                # fallback to saving model weights only
                torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    
    # Save training history
    history_path = f"{config['output']['results_path']}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Save final model
    final_model_path = f"{config['output']['model_save_path']}/final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    print("=== Training Complete ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AI Image Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    try:
        train(config_path=args.config, resume=args.resume)
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        # Write full traceback to a file for post-mortem
        try:
            with open('training_error.log', 'w', encoding='utf-8') as f:
                traceback.print_exc(file=f)
        except Exception:
            pass
        sys.exit(1)
