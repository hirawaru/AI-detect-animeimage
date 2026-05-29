"""
CÁCH 3: Gọi Hugging Face Inference API — không cần cài PyTorch, không cần server.
──────────────────────────────────────────────────────────────────────────────────
- HF chạy model trên server của họ, bạn chỉ gửi HTTP request.
- Cần: HF Access Token (free tier có giới hạn request/tháng)
- Cần cài: pip install requests   (hoặc pip install huggingface_hub)
- Phù hợp khi: prototype nhanh, ứng dụng không muốn cài PyTorch,
  hoặc muốn dùng từ bất kỳ ngôn ngữ nào.

LƯU Ý QUAN TRỌNG:
  HF Inference API chỉ hoạt động tốt với model đã đăng ký pipeline
  (transformers, diffusers...). Model PyTorch thuần (.pth) cần dùng
  "Custom Inference Endpoint" (trả phí) hoặc deploy lên HF Spaces (miễn phí).
  
  → Giải pháp miễn phí tốt nhất: Deploy lên Hugging Face Spaces (xem bên dưới).
"""

import os
import requests
import base64
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")
REPO_ID  = "hirawaru/animeaidetect"

# ── Option A: Dùng HF Inference API (nếu model hỗ trợ) ──────────────────────
def call_hf_inference_api(image_path: str) -> dict:
    """
    Gọi HF Inference API.
    URL: https://api-inference.huggingface.co/models/<repo_id>
    """
    api_url = f"https://api-inference.huggingface.co/models/{REPO_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = requests.post(api_url, headers=headers, data=image_bytes, timeout=30)

    if response.status_code == 503:
        print("⏳ Model đang khởi động (cold start), thử lại sau 20 giây...")
        import time; time.sleep(20)
        response = requests.post(api_url, headers=headers, data=image_bytes, timeout=60)

    response.raise_for_status()
    return response.json()
    # Trả về list: [{"label": "Natural", "score": 0.13}, {"label": "Synthetic", "score": 0.87}]


# ── Option B: Dùng huggingface_hub InferenceClient (wrapper tiện hơn) ────────
def call_hf_inference_client(image_path: str) -> list:
    """
    Dùng InferenceClient từ huggingface_hub >= 0.16
    pip install huggingface_hub
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=REPO_ID, token=HF_TOKEN)

    with open(image_path, "rb") as f:
        result = client.image_classification(f)
    # result: [ClassificationOutput(label='Synthetic', score=0.87), ...]
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if HF_TOKEN == "hf_YOUR_TOKEN_HERE":
        print("⚠️  Hãy set biến môi trường HF_TOKEN trước:")
        print("   Windows: set HF_TOKEN=hf_xxxx")
        print("   Linux  : export HF_TOKEN=hf_xxxx")
        sys.exit(1)

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    print(f"Gọi HF Inference API cho: {image_path}")
    try:
        result = call_hf_inference_api(image_path)
        print("Kết quả:", result)
    except Exception as e:
        print(f"Lỗi: {e}")
        print("→ Model .pth thuần có thể chưa được HF Inference API hỗ trợ.")
        print("→ Xem hướng dẫn deploy lên HF Spaces bên dưới.")


# ═══════════════════════════════════════════════════════════════════════════════
# GIẢI PHÁP MIỄN PHÍ TỐT NHẤT: DEPLOY LÊN HUGGING FACE SPACES
# ═══════════════════════════════════════════════════════════════════════════════
"""
HF Spaces cho phép deploy ứng dụng web miễn phí, bao gồm FastAPI app của bạn.
Sau khi deploy, bạn có URL public: https://<username>-<space-name>.hf.space

CÁCH DEPLOY:
1. Tạo Space mới tại https://huggingface.co/new-space
   - Space name: animeaidetect-demo
   - SDK: Docker  (vì project dùng Dockerfile)
   - Visibility: Public

2. Push code lên Space (Space là một git repo):
   git remote add space https://huggingface.co/spaces/hirawaru/animeaidetect-demo
   git push space main

3. HF tự build Docker image và deploy.
   URL: https://hirawaru-animeaidetect-demo.hf.space

SAU KHI DEPLOY, gọi API từ bất kỳ đâu:

   # Python
   import requests
   res = requests.post(
       "https://hirawaru-animeaidetect-demo.hf.space/predict",
       files={"file": open("image.jpg", "rb")}
   )
   print(res.json())  # {"label": "Synthetic", "prob": 0.87, "all_probs": [...]}

   # curl
   curl -X POST https://hirawaru-animeaidetect-demo.hf.space/predict \\
        -F "file=@image.jpg"

   # JavaScript fetch
   const form = new FormData();
   form.append('file', fileInput.files[0]);
   const res = await fetch('https://hirawaru-animeaidetect-demo.hf.space/predict', {
       method: 'POST', body: form
   });
   const data = await res.json();

LƯU Ý HF SPACES:
- Free tier: CPU only, 16GB RAM, ngủ sau 48h không dùng (cold start ~30s)
- Upgrade lên Spaces Pro ($9/tháng) để có GPU và không ngủ
- Cần thêm HF_TOKEN vào Spaces Secrets để tải model từ private repo
"""
