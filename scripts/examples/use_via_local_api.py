"""
CÁCH 2: Gọi REST API của web app đang chạy cục bộ (web/app.py).
────────────────────────────────────────────────────────────────
- Cần: web app đang chạy tại http://localhost:8000
  → Chạy trước: uvicorn web.app:app --host 0.0.0.0 --port 8000
- Cần cài: pip install requests
- Phù hợp khi: ứng dụng khác (Node.js, mobile, script khác) muốn dùng
  model mà không cần cài PyTorch.
"""

import requests
import sys
from pathlib import Path

API_BASE = "http://localhost:8000"   # Đổi thành IP/domain nếu deploy lên server


def predict_image(image_path: str) -> dict:
    """Gửi ảnh lên /predict, nhận kết quả JSON."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {image_path}")

    with open(path, "rb") as f:
        # multipart/form-data — đúng format FastAPI File(...)
        response = requests.post(
            f"{API_BASE}/predict",
            files={"file": (path.name, f, "image/jpeg")},
            timeout=30,
        )

    response.raise_for_status()
    return response.json()
    # Trả về: {"label": "Synthetic", "prob": 0.87, "all_probs": [0.13, 0.87]}


def get_training_history() -> dict:
    """Lấy lịch sử training để vẽ biểu đồ."""
    response = requests.get(f"{API_BASE}/training-history", timeout=10)
    response.raise_for_status()
    return response.json()
    # Trả về: {"train_loss": [...], "train_acc": [...], "val_loss": [...], "val_acc": [...]}


def check_health() -> bool:
    """Kiểm tra server có đang chạy không."""
    try:
        r = requests.get(API_BASE, timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Kiểm tra server
    if not check_health():
        print(f"❌ Server không phản hồi tại {API_BASE}")
        print("   Hãy chạy: uvicorn web.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    print(f"✅ Server đang chạy tại {API_BASE}")

    # Predict ảnh
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    try:
        result = predict_image(image_path)
        print(f"\n{'─'*35}")
        print(f"  Ảnh      : {image_path}")
        print(f"  Kết quả  : {result['label']}")
        print(f"  Tin cậy  : {result['prob']:.1%}")
        print(f"  Natural  : {result['all_probs'][0]:.1%}")
        print(f"  Synthetic: {result['all_probs'][1]:.1%}")
        print(f"{'─'*35}")
    except requests.HTTPError as e:
        print(f"❌ Lỗi HTTP {e.response.status_code}: {e.response.text}")
    except FileNotFoundError as e:
        print(f"❌ {e}")


# ── Ví dụ dùng trong ứng dụng khác ──────────────────────────────────────────
"""
# Trong Node.js (dùng fetch hoặc axios):

const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function predict(imagePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));

    const res = await axios.post('http://localhost:8000/predict', form, {
        headers: form.getHeaders()
    });
    return res.data;
    // { label: 'Synthetic', prob: 0.87, all_probs: [0.13, 0.87] }
}

# Trong JavaScript (browser, dùng fetch):

async function predict(file) {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
    });
    return await res.json();
}

# Dùng curl:
# curl -X POST http://localhost:8000/predict -F "file=@your_image.jpg"
"""
