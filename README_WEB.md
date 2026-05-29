AI Image Detection — Web deployment

Quick start (local, without Docker):

1. Activate your venv and install web deps:

```powershell
e:\aidetect\.venv\Scripts\activate
python -m pip install -r requirements-web.txt
```

2. Run the API:

```powershell
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

3. Open http://localhost:8000 in a browser and upload an image.

Docker build (recommended for production):

```bash
docker build -t ai-image-web .
docker run -p 8000:8000 -v $(pwd)/models:/app/models ai-image-web
```

Or use `docker-compose up --build`.

Docker image: NumPy compatibility and rebuild instructions
-------------------------------------------------------

If you see runtime errors about NumPy (e.g. "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"), the cause is precompiled wheels (torch/torchvision) being built against a NumPy ABI older than NumPy 2.x. To avoid this, ensure `numpy<2` is installed before installing `torch`/`torchvision` in the Docker image.

Quick fix (already applied in this repo):

1. Confirm `requirements-web.txt` pins `numpy<2`.
2. Build the image with a clean cache so installation order is respected:

```bash
docker compose build --no-cache web
docker compose up -d
```

This installs the requirements (including `numpy<2`) first, then installs `torch`/`torchvision` with `--no-deps` so pip won't upgrade NumPy to 2.x.

CI: example GitHub Actions workflow
----------------------------------

You can add a CI workflow to automatically build the Docker image on push or pull requests. Example workflow file: `.github/workflows/docker-build.yml` (provided in this repo). It builds the image (without pushing) and runs a short smoke step to verify `numpy` and `torch` import correctly in the built image.

If you'd like, I can also add an integration step that runs a small container smoke-test against the `/predict` endpoint (requires shipping a small sample model or a test-mode flag in `web/app.py`).
