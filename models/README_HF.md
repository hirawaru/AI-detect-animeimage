---
language:
- vi
- en
license: mit
pipeline_tag: image-classification
tags:
- image-classification
- pytorch
- efficientnet
- ai-detection
- synthetic-image-detection
- anime
- custom_code
datasets:
- custom
metrics:
- accuracy
- f1
- roc_auc
model-index:
- name: animeaidetect
  results:
  - task:
      type: image-classification
      name: Image Classification
    metrics:
    - type: accuracy
      value: 0.73
      name: Validation Accuracy (5 epochs)
---

# 🔍 AI Image Detection — Natural vs Synthetic (Anime)

Mô hình phân loại ảnh nhị phân phát hiện ảnh do AI tạo ra (**Synthetic**) so với ảnh do con người vẽ (**Natural**), tập trung vào thể loại ảnh anime/illustration.

> **Mục tiêu:** Bảo vệ quyền sáng tác của các họa sĩ trước làn sóng ảnh AI.

---

## 🚀 Use this model

### Cách 1 — Transformers pipeline (khuyến nghị)

```python
from transformers import pipeline

pipe = pipeline(
    "image-classification",
    model="hirawaru/animeaidetect",
    trust_remote_code=True,   # cần vì dùng custom pipeline
)

# Từ file path
result = pipe("your_image.jpg")

# Từ PIL Image
from PIL import Image
result = pipe(Image.open("your_image.jpg"))

# Từ URL
result = pipe("https://example.com/image.jpg")

print(result)
# [{"label": "Synthetic", "score": 0.87},
#  {"label": "Natural",   "score": 0.13}]
```

```bash
pip install transformers torch torchvision Pillow
```

### Cách 2 — PyTorch thuần (không cần Transformers)

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download

# Tải weights
model_path = hf_hub_download(repo_id="hirawaru/animeaidetect", filename="best_model.pth")

# Dựng kiến trúc
model = models.efficientnet_b3(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, 2),
)
ckpt  = torch.load(model_path, map_location="cpu")
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state)
model.eval()

# Inference
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img   = Image.open("your_image.jpg").convert("RGB")
x     = transform(img).unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]

labels = ["Natural", "Synthetic"]
print(f"Predicted : {labels[probs.argmax()]}")
print(f"Confidence: {probs.max():.1%}")
```

---

## 🏗️ Model Architecture

| Thông số | Giá trị |
|----------|---------|
| **Backbone** | EfficientNet-B3 (pretrained ImageNet) |
| **Classifier Head** | Dropout(0.3) → Linear(1536, 2) |
| **Input Size** | 224×224 (inference) / 384×384 (training) |
| **Output Classes** | 2: `Natural` (0), `Synthetic` (1) |
| **Parameters** | ~12M |
| **Framework** | PyTorch 2.1.2 |

---

## 🌐 Web Demo

```bash
git clone https://github.com/<your-username>/aidetect
cd aidetect
pip install -r requirements-web.txt
uvicorn web.app:app --host 0.0.0.0 --port 8000
# Truy cập: http://localhost:8000
```

---

## 📦 Dataset

| Tập | Natural | Synthetic | Tổng |
|-----|---------|-----------|------|
| Train (70%) | ~2,293 | ~2,288 | ~4,581 |
| Validation (15%) | ~491 | ~490 | ~981 |
| Test (15%) | ~491 | ~490 | ~981 |
| **Tổng** | **~3,275** | **~3,268** | **~6,543** |

- **Nguồn:** Kaggle — ảnh anime/illustration
- **Nhãn:** `Natural` = ảnh do họa sĩ vẽ tay, `Synthetic` = ảnh do AI tạo (GAN, Stable Diffusion, DALL-E, Midjourney)
- **Cân bằng lớp:** ~50/50

---

## ⚙️ Training Details

| Hyperparameter | Giá trị |
|----------------|---------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Scheduler | CosineAnnealingLR |
| Early Stopping | patience=10 |
| Max Epochs | 50 |
| Image Size (train) | 384×384 |

**Augmentation:** RandomResizedCrop · RandomHorizontalFlip · RandomVerticalFlip · RandomRotation · ColorJitter · RandomAffine

## 📈 Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.7219 | 51.89% | 0.7166 | 54.53% |
| 2 | 0.6606 | 59.90% | 0.6992 | 66.30% |
| 3 | 0.6475 | 61.06% | 0.5960 | 66.44% |
| 4 | 0.6204 | 65.69% | 0.5679 | 72.78% |
| 5 | 0.5947 | 68.65% | 0.5417 | 73.06% |

> Best Val Accuracy: **73.06%** | Best Val Loss: **0.5417** | Epochs ran: **5** *(đang tiếp tục training)*

---

## ⚠️ Limitations

- Chỉ hoạt động tốt với **ảnh anime/illustration**, không phù hợp cho ảnh chụp thực tế.
- Hiệu suất có thể giảm với các mô hình AI sinh ảnh mới nhất.
- Không xác định được mô hình AI cụ thể nào đã tạo ra ảnh.

---

## 📄 License

MIT — Phục vụ mục đích nghiên cứu và bảo vệ quyền sáng tác.
