"""
Model architectures for AI Image Detection
"""

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(model_name='efficientnet_b3', num_classes=2, pretrained=True, dropout=0.3):
    """
    Create a model for image classification
    
    Args:
        model_name (str): Name of model architecture
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout probability
    
    Returns:
        nn.Module: Model instance
    """
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # Modify final layer
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=pretrained)
        # Modify classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(pretrained=pretrained)
        # Modify head
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


class AIImageClassifier(nn.Module):
    """Wrapper for AI Image classification model"""
    
    def __init__(self, model_name='efficientnet_b3', num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        self.model = create_model(model_name, num_classes, pretrained, dropout)
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except classifier"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier layers
        if hasattr(self.model, 'fc'):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'heads'):
            for param in self.model.heads.parameters():
                param.requires_grad = True
    
    def unfreeze_backbone(self, num_layers=None):
        """Unfreeze backbone layers (for fine-tuning)"""
        for param in self.model.parameters():
            param.requires_grad = True
