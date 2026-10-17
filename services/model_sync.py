"""Sync a model directory from a MinIO bucket down to local disk.

Per ADR-006, workers are meant to load weights from a local cache
(`/model-cache`, a Docker volume) with a fallback to download-on-first-use so
the model persists across container restarts. That fallback wasn't actually
implemented anywhere yet — this is it. Any worker whose model lives as raw
files in a MinIO bucket (not MLflow-registered) can call `sync_if_missing`
before loading.
"""

import logging
from pathlib import Path

from minio import Minio

logger = logging.getLogger(__name__)


def sync_if_missing(
    local_dir: str,
    bucket: str,
    prefix: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
) -> Path:
    """Download every object under `prefix` in `bucket` into `local_dir`.

    No-op if `local_dir` already exists and has at least one file in it —
    that's the "cached from a previous run" case. Not a checksum-verified
    sync: if the bucket contents change, clear the local dir to force a
    re-download.
    """
    dest = Path(local_dir)
    if dest.exists() and any(dest.iterdir()):
        logger.info("Model already cached at %s, skipping MinIO sync", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    client = Minio(endpoint.replace("http://", "").replace("https://", ""), access_key, secret_key, secure=False)

    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    if not objects:
        raise FileNotFoundError(
            f"No objects found under s3://{bucket}/{prefix} — check the prefix "
            f"(e.g. `mc ls sinapse/{bucket}/{prefix}`)."
        )

    logger.info("Syncing %d file(s) from s3://%s/%s to %s", len(objects), bucket, prefix, dest)
    for obj in objects:
        relative = obj.object_name[len(prefix):].lstrip("/")
        if not relative:
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj.object_name, str(target))

    logger.info("Synced model to %s", dest)
    return dest
