import os
from huggingface_hub import HfApi, login

def upload_to_spaces():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        return
    repo_id = "hirawaru/animeaidetect"
    
    login(token=token)
    api = HfApi()
    
    print(f"Creating/Verifying Space: {repo_id}...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
    except Exception as e:
        print(f"Note: {e}")

    # Files to upload
    files_to_upload = [
        "Dockerfile",
        "requirements-web.txt",
        ".dockerignore",
    ]
    
    # Folders to upload (recursive)
    folders_to_upload = [
        "src",
        "web",
        "results", # To show the charts in the demo
    ]

    print("Uploading files...")
    for file in files_to_upload:
        if os.path.exists(file):
            print(f"  Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="space"
            )

    for folder in folders_to_upload:
        if os.path.exists(folder):
            print(f"  Uploading folder {folder}...")
            api.upload_folder(
                folder_path=folder,
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="space"
            )

    print(f"\n🚀 Done! Your Space is building at: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    upload_to_spaces()
