#!/usr/bin/env bash
set -euo pipefail

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"

echo "Waiting for MinIO at ${MINIO_ENDPOINT}..."
until mc alias set sinapse "${MINIO_ENDPOINT}" "${MINIO_USER}" "${MINIO_PASSWORD}" > /dev/null 2>&1; do
  sleep 2
done
echo "Connected to MinIO."

create_bucket() {
  local bucket="$1"
  if mc ls "sinapse/${bucket}" > /dev/null 2>&1; then
    echo "Bucket '${bucket}' already exists, skipping."
  else
    mc mb "sinapse/${bucket}"
    echo "Bucket '${bucket}' created."
  fi
}

echo "Creating buckets..."
create_bucket bronze
create_bucket mlflow-artifacts
create_bucket models

echo "Setting bucket policies..."
mc anonymous set private sinapse/bronze
mc anonymous set private sinapse/mlflow-artifacts
mc anonymous set download sinapse/models

echo ""
echo "Bucket initialization complete"
echo "  bronze/          → BOs, inquéritos (private, append-only)"
echo "  mlflow-artifacts/ → MLflow experiment artifacts (private)"
echo "  models/          → Versioned model weights (download = workers read-only)"
echo ""
