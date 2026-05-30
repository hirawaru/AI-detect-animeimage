import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os
import io
import base64

class EndpointHandler:
    """
    Custom handler for Hugging Face Inference API.
    This class loads the EfficientNet-B3 model and performs inference.
    """
    def __init__(self, path=""):
        # Load model architecture (EfficientNet-B3)
        self.model = models.efficientnet_b3(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.classifier[1].in_features, 2),
        )
        
        # Load weights
        model_path = os.path.join(path, "best_model.pth")
        if not os.path.exists(model_path):
            # In some environments, path might be the directory containing the file
            model_path = os.path.join(path, "best_model.pth")
            
        checkpoint = torch.load(model_path, map_location="cpu")
        # Handle both state_dict only and full checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Define transforms (Standard ImageNet normalization for EfficientNet)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.labels = ["Natural", "Synthetic"]

    def __call__(self, data):
        """
        Args:
            data (:obj:`dict`):
                data contains the raw data and some parameters.
                "inputs" should contain the image data.
        Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        inputs = data.pop("inputs", data)
        
        # Handle image data (can be bytes, PIL, or base64 string)
        if isinstance(inputs, bytes):
            image = Image.open(io.BytesIO(inputs)).convert("RGB")
        elif isinstance(inputs, str):
            # Assume base64
            image = Image.open(io.BytesIO(base64.b64decode(inputs))).convert("RGB")
        elif isinstance(inputs, Image.Image):
            image = inputs.convert("RGB")
        else:
            # Try to treat it as a dict or other format if needed
            raise ValueError(f"Unsupported input type: {type(inputs)}")

        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
        # Format results as a list of dicts (label, score)
        results = []
        for i, label in enumerate(self.labels):
            results.append({
                "label": label,
                "score": float(probs[i])
            })
            
        # Sort by score descending (highest confidence first)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
