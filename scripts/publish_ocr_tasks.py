#!/usr/bin/env python3
"""Publish OCR tasks to Redis for every image in tests/fixtures/images/.

Each task uses image_path so the worker reads the file from a volume-mounted
directory inside the container.

Usage:
    python scripts/publish_ocr_tasks.py [--redis URL] [--stream STREAM] [--path-prefix PREFIX]

Examples:
    # Default: localhost:6380, stream ocr:tasks, path /fixtures/images/
    python publish_ocr_tasks.py

    # Custom Redis
    python publish_ocr_tasks.py --redis redis://user:pass@host:6379/0
"""

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "images"


def main():
    parser = argparse.ArgumentParser(description="Publish OCR tasks to Redis")
    parser.add_argument("--redis", default="redis://localhost:6380", help="Redis URL")
    parser.add_argument("--stream", default="ocr:tasks", help="Redis stream name")
    parser.add_argument(
        "--path-prefix",
        default="/fixtures/images",
        help="Path prefix for image_path inside container",
    )
    args = parser.parse_args()

    try:
        from redis import Redis
    except ImportError:
        print("redis not installed. Run: pip install redis")
        sys.exit(1)

    images = sorted(
        p for p in FIXTURE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        print(f"No image files found in {FIXTURE_DIR}")
        print("Run download_test_images.py first, or place images manually.")
        return

    r = Redis.from_url(args.redis, decode_responses=True)

    for img in images:
        task = {
            "document_id": img.stem,
            "image_path": f"{args.path_prefix}/{img.name}",
        }
        msg_id = r.xadd(args.stream, task, maxlen=1000)
        print(f"  Published {img.name} -> {args.stream} [{msg_id}]")

    print(f"\nDone. {len(images)} task(s) published to {args.stream}")
    print(f"Worker will read images from: {args.path_prefix}")

    if r.ping():
        print("Redis connected OK")
    else:
        print("WARNING: Redis ping failed")


if __name__ == "__main__":
    import sys
    main()
