"""
scripts/download_weights.py
===========================
Downloads PACE model weights from the GitHub release if they are not
already present locally. Safe to run repeatedly — skips files that exist.

Usage:
    python scripts/download_weights.py          # download if missing
    python scripts/download_weights.py --force  # re-download even if present

Called automatically by the app at startup via pages/loading.py when
is_pace_model_ready() returns False.
"""

import argparse
import os
import sys
import urllib.request

# ── Release configuration ──────────────────────────────────────────────────────
REPO        = "Spring-ISYS-2026-Alpha-Team/Accessorial_Cost_Detection_Engine"
RELEASE_TAG = "v1.0-weights"
BASE_URL    = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

MODEL_FILES = {
    "models/pace_transformer_weights.pt": f"{BASE_URL}/pace_transformer_weights.pt",
    "models/pace_transformer_ltl.pt":     f"{BASE_URL}/pace_transformer_ltl.pt",
    "models/artifacts_ltl.pkl":           f"{BASE_URL}/artifacts_ltl.pkl",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_weights(force: bool = False) -> bool:
    """
    Download missing model weight files from the GitHub release.
    Stdout-safe: works in both terminal and Streamlit Cloud environments.
    Returns True if all files are present after the attempt.
    """
    all_ok = True

    for rel_path, url in MODEL_FILES.items():
        local_path = os.path.join(REPO_ROOT, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path) and not force:
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"  ✓ {rel_path} already present ({size_mb:.1f} MB) — skipping")
            continue

        print(f"  ↓ Downloading {rel_path} from GitHub release {RELEASE_TAG} ...")
        try:
            if not url.startswith("https://"):
                raise ValueError(f"Refusing to download from non-HTTPS URL: {url}")
            # Use a simple request without a progress hook — stdout.write/flush
            # raises in Streamlit Cloud's managed environment and aborts the download.
            urllib.request.urlretrieve(url, local_path)  # nosec B310
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"    ✓ Saved to {local_path} ({size_mb:.1f} MB)")
        except Exception as e:
            # Clean up any partial file so a retry starts fresh
            if os.path.exists(local_path):
                os.remove(local_path)
            print(f"    ✗ Failed to download {rel_path}: {e}")
            print(f"      URL attempted: {url}")
            all_ok = False

    return all_ok


def ensure_weights_ready() -> bool:
    """
    Called at app startup. Downloads weights only if missing.
    Returns True when both weight files are present and ready.
    """
    from pipeline.config import is_pace_model_ready
    if is_pace_model_ready():
        return True
    print("PACE model weights not found — downloading from GitHub release...")
    ok = download_weights(force=False)
    if ok:
        print("PACE model weights ready.")
    else:
        print(
            "WARNING: Could not download model weights. "
            f"Check that the release '{RELEASE_TAG}' exists and assets are attached at: "
            f"https://github.com/{REPO}/releases"
        )
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PACE model weights from GitHub release")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist locally"
    )
    args = parser.parse_args()

    print(f"PACE Weight Downloader")
    print(f"  Repo:    {REPO}")
    print(f"  Release: {RELEASE_TAG}")
    print(f"  Files:   {', '.join(MODEL_FILES.keys())}")
    print()

    ok = download_weights(force=args.force)
    sys.exit(0 if ok else 1)