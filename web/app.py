#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from src.model import AIImageClassifier
import os
import json

app = FastAPI(title="AI Image Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 224))
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_PATH = os.environ.get("MODEL_PATH", "models/checkpoints/best_model.pth")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "hirawaru/animeaidetect")  # Default to the newly uploaded model
MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet_b3")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", 2))
LABELS = os.environ.get("LABELS", "Natural,Synthetic").split(",")
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 5 * 1024 * 1024))  # 5MB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = AIImageClassifier(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
    
    path = MODEL_PATH
    # Nếu file cục bộ không tồn tại và có cấu hình HF_REPO_ID, tải tự động từ Hugging Face Hub
    if not os.path.exists(path) and HF_REPO_ID:
        print(f"File model cục bộ '{path}' không tồn tại. Đang tải từ Hugging Face Hub: {HF_REPO_ID}...")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=HF_REPO_ID, filename="best_model.pth")
            print(f"Đã tải thành công model từ Hugging Face về cache: {path}")
        except Exception as e:
            print(f"Lỗi khi tải model từ Hugging Face Hub: {e}")
            print("Sẽ cố gắng thử load từ MODEL_PATH cục bộ mặc định...")

    try:
        ckpt = torch.load(path, map_location=device)
        # Support full checkpoint dicts saved from training
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt
        else:
            state = ckpt

        try:
            model.load_state_dict(state)
            print(f"Loaded model weights from {path}")
        except RuntimeError:
            # Try stripping possible 'module.' prefixes from DataParallel/Distributed wrappers
            stripped = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(stripped)
            print(f"Loaded model weights from {path} (stripped 'module.' prefix)")
    except Exception as e:
        print("Warning: failed to load model:", e)
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="<h3>AI Image Detection API</h3>", status_code=200)
    html = open(html_path, "r", encoding="utf-8").read()
    return HTMLResponse(content=html, status_code=200)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy().tolist()
        idx = int(torch.argmax(torch.tensor(probs)).item())

    return {"label": LABELS[idx], "prob": float(probs[idx]), "all_probs": [float(p) for p in probs]}


@app.get("/training-history")
async def training_history():
    """Trả về lịch sử huấn luyện để vẽ biểu đồ trên web."""
    history_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "training_history.json")
    if not os.path.exists(history_path):
        raise HTTPException(status_code=404, detail="training_history.json not found")
    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.get("/results/{filename}")
async def serve_result_image(filename: str):
    """Serve ảnh kết quả (confusion matrix, ROC curve, EDA plots, ...)."""
    # Chỉ cho phép file ảnh, tránh path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".svg"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="File type not allowed")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    file_path = os.path.join(results_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return FileResponse(file_path, media_type=f"image/{ext.lstrip('.')}")
