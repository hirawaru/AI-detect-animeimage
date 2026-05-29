"""
CÁCH 1: Kéo file .pth từ Hugging Face Hub về, chạy inference cục bộ.
─────────────────────────────────────────────────────────────────────
- Cần cài: pip install torch torchvision huggingface_hub Pillow
- File .pth được cache tại ~/.cache/huggingface/hub/
  → Lần đầu tải (~50MB), các lần sau dùng cache, không tải lại.
- Phù hợp khi: ứng dụng Python, cần tốc độ cao, chạy offline sau lần đầu.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download

REPO_ID = "hirawaru/animeaidetect"
LABELS  = ["Natural", "Synthetic"]


# ── Bước 1: Tải model từ HF Hub (tự động cache) ──────────────────────────────
def load_model(device: str = "cpu"):
    # Tải file .pth về cache (~/.cache/huggingface/hub/)
    model_path = hf_hub_download(repo_id=REPO_ID, filename="best_model.pth")
    print(f"Model cached tại: {model_path}")

    # Dựng lại kiến trúc EfficientNet-B3
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 2),
    )

    # Load weights
    ckpt  = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)  # hỗ trợ cả 2 định dạng checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ── Bước 2: Transform ảnh ────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ── Bước 3: Predict ──────────────────────────────────────────────────────────
def predict(image_path: str, model, device: str = "cpu") -> dict:
    img = Image.open(image_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()

    idx = probs.argmax().item()
    return {
        "label"     : LABELS[idx],
        "confidence": round(probs[idx].item(), 4),
        "Natural"   : round(probs[0].item(), 4),
        "Synthetic" : round(probs[1].item(), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model  = load_model(device)
    result = predict(image_path, model, device)

    print(f"\n{'─'*35}")
    print(f"  Ảnh      : {image_path}")
    print(f"  Kết quả  : {result['label']}")
    print(f"  Tin cậy  : {result['confidence']:.1%}")
    print(f"  Natural  : {result['Natural']:.1%}")
    print(f"  Synthetic: {result['Synthetic']:.1%}")
    print(f"{'─'*35}")
