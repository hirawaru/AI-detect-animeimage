"""
Custom pipeline cho Hugging Face — cho phép dùng:
    from transformers import pipeline
    pipe = pipeline("image-classification", model="hirawaru/animeaidetect",
                    trust_remote_code=True)
    result = pipe("image.jpg")
"""

from typing import Union
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from transformers import Pipeline


# ── Kiến trúc model ───────────────────────────────────────────────────────────
def _build_efficientnet_b3(num_labels: int = 2, dropout: float = 0.3):
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier[1].in_features, num_labels),
    )
    return model


# ── Transform ─────────────────────────────────────────────────────────────────
_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ── Custom Pipeline ───────────────────────────────────────────────────────────
class AIImageDetectionPipeline(Pipeline):
    """
    Pipeline phát hiện ảnh AI (Natural vs Synthetic).

    Ví dụ sử dụng:
        from transformers import pipeline

        pipe = pipeline(
            "image-classification",
            model="hirawaru/animeaidetect",
            trust_remote_code=True,
        )

        # Từ file path
        result = pipe("image.jpg")

        # Từ PIL Image
        from PIL import Image
        result = pipe(Image.open("image.jpg"))

        # Từ URL
        result = pipe("https://example.com/image.jpg")

        # Kết quả:
        # [{"label": "Synthetic", "score": 0.87},
        #  {"label": "Natural",   "score": 0.13}]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, image):
        """Nhận PIL Image / path / URL, trả tensor."""
        if isinstance(image, (str, Path)):
            path = str(image)
            if path.startswith("http://") or path.startswith("https://"):
                import requests
                from io import BytesIO
                response = requests.get(path, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(path).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported input type: {type(image)}")

        tensor = _TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224)
        return {"pixel_values": tensor}

    def _forward(self, model_inputs):
        pixel_values = model_inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values)
            probs  = torch.softmax(logits, dim=1)[0].cpu()
        return {"probs": probs}

    def postprocess(self, model_outputs):
        probs  = model_outputs["probs"]
        labels = self.model.config.id2label if hasattr(self.model, "config") else {0: "Natural", 1: "Synthetic"}

        results = [
            {"label": labels[i], "score": round(probs[i].item(), 4)}
            for i in range(len(probs))
        ]
        # Sắp xếp theo score giảm dần (giống chuẩn HF)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
