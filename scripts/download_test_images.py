#!/usr/bin/env python3
"""Download test images from Google Drive for OCR testing.

Usage:
    pip install gdown
    python scripts/download_test_images.py <google-drive-url-or-id>

Examples:
    # Download a single file by URL
    python download_test_images.py "https://drive.google.com/file/d/1ABC123/view"

    # Download a whole folder
    python download_test_images.py "https://drive.google.com/drive/folders/1XYZ789"
"""

import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "images"


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_test_images.py <google-drive-url-or-id>")
        sys.exit(1)

    try:
        import gdown
    except ImportError:
        print("gdown not installed. Run: pip install gdown")
        sys.exit(1)

    url_or_id = sys.argv[1]
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(FIXTURE_DIR.iterdir())
    if existing:
        print(f"  Directory already has {len(existing)} file(s). Delete them first or use a fresh directory.")
        resp = input("  Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)
        for f in existing:
            f.unlink()

    print(f"Downloading to {FIXTURE_DIR} ...")

    if "/drive/folders/" in url_or_id or "/folders/" in url_or_id:
        gdown.download_folder(url_or_id, output=str(FIXTURE_DIR), quiet=False)
    else:
        gdown.download(url_or_id, output=str(FIXTURE_DIR), quiet=False)

    print("\nDone. Files downloaded:")
    for f in sorted(FIXTURE_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
