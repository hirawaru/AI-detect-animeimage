# AI-Generated Image Detection Model

Một model deep learning để phát hiện ảnh do AI tạo ra (Synthetic) và ảnh thực (Natural) nhằm bảo vệ quyền sáng tác của các họa sĩ.

## 📊 Dataset

- **Natural images**: ~3,275 ảnh thực chất lượng cao
- **Synthetic images**: ~3,268 ảnh do AI tạo ra (GAN, Stable Diffusion, DALL-E, v.v.)
- **Total**: ~6,543 ảnh được cân bằng tốt

## 🏗️ Project Structure

```
aidetect/
├── data/                  # Thư mục dữ liệu
│   ├── train/            # Dữ liệu huấn luyện
│   ├── val/              # Dữ liệu validation
│   └── test/             # Dữ liệu test
├── src/                  # Source code
│   ├── __init__.py
│   ├── dataset.py        # Data loading
│   ├── model.py          # Model architecture
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation metrics
│   └── inference.py      # Inference pipeline
├── notebooks/            # Jupyter notebooks
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   └── 02_results_analysis.ipynb       # Results visualization
├── scripts/              # Utility scripts
│   ├── prepare_data.py   # Split data into train/val/test
│   └── download_data.py  # Download from sources
├── models/               # Trained models
│   └── checkpoints/      # Model checkpoints
├── results/              # Training results
│   ├── logs/
│   ├── metrics.json
│   └── confusion_matrix.png
├── config.yaml          # Configuration file
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Split raw data into train/val/test
python scripts/prepare_data.py
```

### 3. Train Model

```bash
# Train the model
python src/train.py
```

### 4. Evaluate Model

```bash
# Evaluate on test set
python src/evaluate.py
```

### 5. Run Inference

```bash
# Predict on new images
python src/inference.py --image_path <path_to_image>
```

## 📓 Notebooks

- **01_eda.ipynb**: Khám phá dữ liệu, thống kê ảnh, phân tích tính năng
- **02_results_analysis.ipynb**: Visualize kết quả, confusion matrix, ROC curves

## 🔧 Configuration

Chỉnh sửa `config.yaml` để:
- Thay đổi paths dữ liệu
- Chọn model architecture (ResNet50, EfficientNet, Vision Transformer)
- Điều chỉnh hyperparameters (learning rate, batch size, epochs)
- Cấu hình augmentation

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1-Score | - |
| ROC-AUC | - |

*(Sẽ cập nhật sau huấn luyện)*

## 🎯 Models Supported

- **ResNet50**: Nhanh, khả năng tổng quát hóa tốt
- **EfficientNet B3**: Cân bằng tốt giữa accuracy và tốc độ
- **Vision Transformer (ViT)**: State-of-the-art accuracy nhưng cần GPU mạnh

## 💾 Checkpointing

Model checkpoints được lưu tự động trong `models/checkpoints/` nếu validation loss cải thiện.

## 🔍 Features

- ✅ Transfer learning với pretrained models
- ✅ Data augmentation
- ✅ Early stopping
- ✅ Tensorboard logging
- ✅ Confusion matrix & ROC curves
- ✅ Inference pipeline
- ✅ Model export (ONNX support)
- ✅ **Hugging Face Inference API support**

## 🌐 Hugging Face Integration

Mô hình đã được upload lên Hugging Face Hub: [hirawaru/animeaidetect](https://huggingface.co/hirawaru/animeaidetect)

### Sử dụng Hugging Face Inference API (Không cần cài đặt)

Bạn có thể gọi mô hình từ bất kỳ đâu qua API của Hugging Face:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/hirawaru/animeaidetect"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

print(query("test.jpg"))
```

### Sử dụng với thư viện Transformers

```python
from transformers import pipeline

pipe = pipeline("image-classification", model="hirawaru/animeaidetect", trust_remote_code=True)
result = pipe("image.jpg")
print(result)
```

## 📝 License

Phục vụ mục đích bảo vệ quyền sáng tác.

## 👥 Author

Created for AI-generated content detection project
