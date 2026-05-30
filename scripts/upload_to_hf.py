#!/usr/bin/env python3
"""
Upload model lên Hugging Face Hub kèm:
  - best_model.pth        (weights)
  - config.json           (khai báo pipeline → nút "Use this model")
  - preprocessor_config.json
  - pipeline.py           (custom pipeline cho transformers)
  - README.md             (Model Card)

Chạy: python scripts/upload_to_hf.py
"""
import os
import sys
import json
from pathlib import Path
from huggingface_hub import HfApi, login

DEFAULT_MODEL_PATH = "models/checkpoints/best_model.pth"
DEFAULT_REPO_ID    = "hirawaru/animeaidetect"


def load_training_history() -> dict | None:
    path = Path("results/training_history.json")
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--token", type=str, help="Hugging Face Access Token")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Repo ID (e.g. user/model)")
    args = parser.parse_args()

    print("=" * 55)
    print("🚀  UPLOAD MODEL LÊN HUGGING FACE MODEL HUB")
    print("=" * 55)

    # ── Kiểm tra file model ───────────────────────────────────────────────────
    model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model tại: {model_path}")
        sys.exit(1)
    print(f"✅ Tìm thấy model: {model_path}")

    # ── Token ─────────────────────────────────────────────────────────────────
    token = args.token or os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("\nLấy token tại: https://huggingface.co/settings/tokens")
        print("(Chọn quyền WRITE khi tạo token)")
        token = input("Nhập Hugging Face Access Token: ").strip()
    if not token:
        print("❌ Token không được để trống!")
        sys.exit(1)

    try:
        login(token=token)
        print("✅ Đăng nhập thành công!")
    except Exception as e:
        print(f"❌ Đăng nhập thất bại: {e}")
        sys.exit(1)

    # ── Repo ID ───────────────────────────────────────────────────────────────
    repo_id = args.repo_id
    if not repo_id:
        print(f"\nRepo ID mặc định: {DEFAULT_REPO_ID}")
        repo_id = input("Nhập Repo ID (Enter để dùng mặc định): ").strip()
        if not repo_id:
            repo_id = DEFAULT_REPO_ID
    
    if "/" not in repo_id:
        print("❌ Repo ID phải có dạng <username>/<repo-name>")
        sys.exit(1)

    api = HfApi()

    # ── Tạo repo ──────────────────────────────────────────────────────────────
    print(f"\n📁 Đang tạo/kiểm tra repository: {repo_id} ...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("✅ Repository sẵn sàng!")
    except Exception as e:
        print(f"⚠️  Không thể tạo repo: {e}")

    # ── Upload model weights ──────────────────────────────────────────────────
    print(f"\n📤 [1/5] Uploading best_model.pth ...")
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.pth",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Upload best_model.pth",
        )
        print("✅ best_model.pth OK")
    except Exception as e:
        print(f"❌ Lỗi upload model: {e}")
        sys.exit(1)

    # ── Upload config files (kích hoạt nút "Use this model → Transformers") ──
    hf_upload_dir = Path("models/hf_upload")
    extra_files = [
        ("config.json",                "config.json"),
        ("preprocessor_config.json",   "preprocessor_config.json"),
        ("pipeline.py",                "pipeline.py"),
        ("handler.py",                 "handler.py"),
    ]

    for i, (local_name, repo_name) in enumerate(extra_files, start=2):
        local_path = hf_upload_dir / local_name
        print(f"\n📤 [{i}/5] Uploading {local_name} ...")
        if not local_path.exists():
            print(f"⚠️  Không tìm thấy {local_path}, bỏ qua.")
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=f"Add {repo_name} for Transformers pipeline support",
            )
            print(f"✅ {local_name} OK")
        except Exception as e:
            print(f"⚠️  Lỗi upload {local_name}: {e}")

    # ── Upload Model Card ─────────────────────────────────────────────────────
    print(f"\n📝 [5/5] Uploading README.md (Model Card) ...")
    card_local = Path("models/README_HF.md")
    if not card_local.exists():
        print(f"⚠️  Không tìm thấy {card_local}")
        print("   Hãy chạy lại sau khi tạo file models/README_HF.md")
    else:
        card_content = card_local.read_text(encoding="utf-8")
        try:
            api.upload_file(
                path_or_fileobj=card_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message="Add Model Card with Transformers pipeline support",
            )
            print("✅ README.md OK")
        except Exception as e:
            print(f"⚠️  Lỗi upload README.md: {e}")
            print(f"   Copy thủ công nội dung từ {card_local} lên HF.")

    # ── Kết quả ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("🎉  HOÀN TẤT!")
    print(f"   Model: https://huggingface.co/{repo_id}")
    print("=" * 55)
    print("\n💡 Dùng qua Transformers:")
    print("```python")
    print("from transformers import pipeline")
    print(f'pipe = pipeline("image-classification", model="{repo_id}", trust_remote_code=True)')
    print('result = pipe("your_image.jpg")')
    print("```")


if __name__ == "__main__":
    main()
