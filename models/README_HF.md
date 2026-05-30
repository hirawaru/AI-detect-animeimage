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
      value: 0.9189
      name: Test Accuracy (20 epochs)
---

# 🔍 AI Image Detection — Natural vs Synthetic (Anime)

Mô hình phân loại ảnh nhị phân phát hiện ảnh do AI tạo ra (**Synthetic**) so với ảnh do con người vẽ (**Natural**), tập trung vào thể loại ảnh anime/illustration.

> **Mục tiêu:** Bảo vệ quyền sáng tác của các họa sĩ trước làn sóng ảnh AI.

---

## ⚡ API Integration (Dành cho nhà phát triển)

Bạn có thể tích hợp mô hình này vào ứng dụng của mình thông qua Hugging Face Inference API mà không cần cài đặt môi trường phức tạp.

### 🐍 Python (Sử dụng requests)
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/hirawaru/animeaidetect"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("test.jpg")
print(output)
```

### 🌐 JavaScript
```javascript
async function query(fileData) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/hirawaru/animeaidetect",
		{
			headers: { Authorization: "Bearer YOUR_HF_TOKEN" },
			method: "POST",
			body: fileData,
		}
	);
	const result = await response.json();
	return result;
}
```

### 💻 cURL
```bash
curl https://api-inference.huggingface.co/models/hirawaru/animeaidetect \
	-X POST \
	--data-binary '@image.jpg' \
	-H "Authorization: Bearer YOUR_HF_TOKEN"
```

---

## 🚀 Use locally with Transformers

```python
from transformers import pipeline

pipe = pipeline(
    "image-classification",
    model="hirawaru/animeaidetect",
    trust_remote_code=True,
)

result = pipe("your_image.jpg")
print(result)
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
| Batch Size | 16 (on CPU) / 32 (on GPU) |
| Scheduler | CosineAnnealingLR |
| Early Stopping | patience=10 |
| Max Epochs | 20 (phiên bản hiện tại) |
| Image Size (train) | 384×384 |

**Augmentation:** RandomResizedCrop · RandomHorizontalFlip · RandomVerticalFlip · RandomRotation · ColorJitter · RandomAffine

## 📈 Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.7219 | 51.89% | 0.7166 | 54.53% |
| 5 | 0.5947 | 68.65% | 0.5417 | 73.06% |
| 10 | 0.2240 | 91.14% | 0.2578 | 90.88% |
| 20 | 0.0508 | 98.30% | 0.5547 | **91.89%** |

> Best Val Accuracy: **91.89%** | Best Val Loss: **0.2578** (at Epoch 10) | Epochs ran: **20**

---

## 📊 Test Set Metrics

- **Accuracy:** 91.89%
- **F1-Score:** 91.18%
- **ROC-AUC:** 97.37%

---

## ⚠️ Limitations

- Chỉ hoạt động tốt với **ảnh anime/illustration**, không phù hợp cho ảnh chụp thực tế.
- Hiệu suất có thể giảm với các mô hình AI sinh ảnh mới nhất.
- Không xác định được mô hình AI cụ thể nào đã tạo ra ảnh.

---

## 📄 License

MIT — Phục vụ mục đích nghiên cứu và bảo vệ quyền sáng tác.

---

## ⚠️ Limitations

- Chỉ hoạt động tốt với **ảnh anime/illustration**, không phù hợp cho ảnh chụp thực tế.
- Hiệu suất có thể giảm với các mô hình AI sinh ảnh mới nhất.
- Không xác định được mô hình AI cụ thể nào đã tạo ra ảnh.

---

## 📄 License

MIT — Phục vụ mục đích nghiên cứu và bảo vệ quyền sáng tác.
