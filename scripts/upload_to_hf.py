"""
Upload danAI-55M-Reasoning to Hugging Face Hub.

Usage:
    ./venv/bin/python scripts/upload_to_hf.py [--token YOUR_HF_WRITE_TOKEN]
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import HfApi, login


def upload_danai_to_hf(repo_id: str = "asjadilahi/danAI-55M-Reasoning", folder_path: str = "hf_export", token: str = None):
    print("=" * 80)
    print("  🚀 UPLOADING danAI-55M-Reasoning TO HUGGING FACE HUB")
    print(f"  • Target Repo: https://huggingface.co/{repo_id}")
    print(f"  • Local Directory: {folder_path}")
    print("=" * 80)

    if token:
        login(token=token)

    api = HfApi(token=token)

    # 1. Create repo if not exists
    print("\n[1/2] Creating/verifying repository on Hugging Face...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"  ✓ Repository verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  Note on create_repo: {e}")

    # 2. Upload folder
    print("\n[2/2] Uploading model weights, configs, logo, and README...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        commit_message="Initial release of danAI-55M-Reasoning (54.5M SLM with Native Agentic Tools and <think> CoT)",
    )

    print("\n" + "=" * 80)
    print(f"🎉 SUCCESS! danAI-55M-Reasoning is now live on Hugging Face!")
    print(f"🔗 URL: https://huggingface.co/{repo_id}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload danAI model to Hugging Face")
    parser.add_argument("--repo-id", type=str, default="asjadilahi/danAI-55M-Reasoning", help="Hugging Face repo ID")
    parser.add_argument("--folder", type=str, default="hf_export", help="Path to export directory")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (optional if already logged in)")
    args = parser.parse_args()

    upload_danai_to_hf(repo_id=args.repo_id, folder_path=args.folder, token=args.token)
