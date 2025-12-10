#!/usr/bin/env python3
"""
Download required models from HuggingFace.

This script downloads:
1. Fine-tuned sbert model from garrykuwanto/mechanics-retrieval
2. LFM2 GGUF model (Q4_0 quantized version) from LiquidAI/LFM2-1.2B-RAG-GGUF
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download


def download_sbert_model(output_dir: str = "models/sbert") -> None:
    """Download the fine-tuned sbert model."""
    print("=" * 80)
    print("Downloading fine-tuned sbert model...")
    print(f"Repository: garrykuwanto/mechanics-retrieval")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download entire model repository
    try:
        snapshot_download(
            repo_id="garrykuwanto/mechanics-retrieval",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Successfully downloaded sbert model to {output_dir}")
    except Exception as e:
        print(f"✗ Error downloading sbert model: {e}")
        raise


def download_lfm2_model(output_dir: str = "models/lfm2") -> None:
    """Download the LFM2 GGUF model (Q4_0 quantized version only)."""
    print("\n" + "=" * 80)
    print("Downloading LFM2 model (Q4_0 quantized)...")
    print(f"Repository: LiquidAI/LFM2-1.2B-RAG-GGUF")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download only the Q4_0 GGUF file
    filename = "LFM2-1.2B-RAG-Q4_0.gguf"
    try:
        downloaded_path = hf_hub_download(
            repo_id="LiquidAI/LFM2-1.2B-RAG-GGUF",
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Successfully downloaded {filename} to {output_dir}")
        print(f"  File path: {downloaded_path}")
    except Exception as e:
        print(f"✗ Error downloading LFM2 model: {e}")
        raise


def check_models_exist(sbert_dir: str, lfm2_dir: str) -> tuple[bool, bool]:
    """Check if models already exist."""
    sbert_exists = (
        Path(sbert_dir).exists()
        and Path(sbert_dir).is_dir()
        and any(Path(sbert_dir).iterdir())
    )
    lfm2_exists = Path(lfm2_dir).joinpath("LFM2-1.2B-RAG-Q4_0.gguf").exists()

    return sbert_exists, lfm2_exists


def main() -> None:
    """Main function to download models."""
    parser = argparse.ArgumentParser(
        description="Download required models from HuggingFace"
    )
    parser.add_argument(
        "--sbert-dir",
        type=str,
        default="models/sbert",
        help="Directory to save sbert model (default: models/sbert)",
    )
    parser.add_argument(
        "--lfm2-dir",
        type=str,
        default="models/lfm2",
        help="Directory to save LFM2 model (default: models/lfm2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if models already exist",
    )
    parser.add_argument(
        "--sbert-only",
        action="store_true",
        help="Only download sbert model",
    )
    parser.add_argument(
        "--lfm2-only",
        action="store_true",
        help="Only download LFM2 model",
    )

    args = parser.parse_args()

    # Check which models to download
    download_sbert = not args.lfm2_only
    download_lfm2 = not args.sbert_only

    # Check if models already exist
    sbert_exists, lfm2_exists = check_models_exist(args.sbert_dir, args.lfm2_dir)

    if not args.force:
        if sbert_exists and download_sbert:
            print(f"⚠ sbert model already exists at {args.sbert_dir}")
            print("  Use --force to re-download")
            download_sbert = False

        if lfm2_exists and download_lfm2:
            print(f"⚠ LFM2 model already exists at {args.lfm2_dir}")
            print("  Use --force to re-download")
            download_lfm2 = False

    # Download models
    if download_sbert:
        download_sbert_model(args.sbert_dir)

    if download_lfm2:
        download_lfm2_model(args.lfm2_dir)

    # Summary
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    if download_sbert:
        print(f"✓ sbert model: {args.sbert_dir}")
    else:
        print(f"○ sbert model: {args.sbert_dir} (skipped)")

    if download_lfm2:
        print(f"✓ LFM2 model: {args.lfm2_dir}/LFM2-1.2B-RAG-Q4_0.gguf")
    else:
        print(f"○ LFM2 model: {args.lfm2_dir}/LFM2-1.2B-RAG-Q4_0.gguf (skipped)")

    print("=" * 80)
    print("\nAll models ready! You can now run the evaluation pipeline.")


if __name__ == "__main__":
    main()
