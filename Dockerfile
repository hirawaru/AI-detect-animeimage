FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libjpeg-dev && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-web.txt
# Install PyTorch CPU wheels without pulling dependencies (dependencies installed above)
RUN pip install --no-cache-dir --no-deps torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html || true

COPY . /app

# HF Spaces yêu cầu chạy với user non-root (uid=1000)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
