import io
import logging

from minio import Minio

logger = logging.getLogger(__name__)


def build_minio_client(endpoint: str, access_key: str, secret_key: str) -> Minio:
    """The one Minio client constructor in this package.

    Minio wants host[:port], but endpoints arrive from env/compose with a
    scheme often enough that both call sites used to strip it themselves —
    and storage.py stripped only `http://`, so an `https://minio:9000`
    reached the client intact and every request URL came out as
    `http://https://minio:9000/...`. The scheme is stripped, not used to
    turn TLS on: callers have always run `secure=False`.
    """
    return Minio(
        endpoint.replace("http://", "").replace("https://", ""),
        access_key,
        secret_key,
        secure=False,
    )


class StorageService:
    """Salva resultados de workers no MinIO (bucket bronze)."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self._client = build_minio_client(endpoint, access_key, secret_key)
        self._bucket = bucket

    def load(self, prefix: str, document_id: str) -> str | None:
        path = f"ml-results/{prefix}/{document_id}"
        try:
            resp = self._client.get_object(self._bucket, path)
            return resp.read().decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to load %s from MinIO: %s", path, exc)
            return None

    def save(
        self,
        prefix: str,
        document_id: str,
        data: bytes,
        content_type: str = "text/plain",
    ) -> str:
        """Write an object to bronze.

        Raises on failure rather than returning None: the caller notifies the
        backend about this text, and a swallowed write means the text lives
        only in Redis — never in bronze — with the task already acked and
        never retried. Same argument download() makes for reads.
        """
        path = f"ml-results/{prefix}/{document_id}"
        try:
            self._client.put_object(
                self._bucket,
                path,
                io.BytesIO(data),
                len(data),
                content_type=content_type,
            )
            logger.info("Saved %s to %s/%s", path, self._bucket, path)
            return path
        except Exception as exc:
            logger.warning("Failed to save %s to MinIO: %s", path, exc)
            raise

    def download(self, bucket: str, key: str) -> bytes:
        """Fetch an arbitrary object by explicit bucket/key (e.g. a source PDF
        the backend uploaded — a different bucket/path convention than this
        service's own `ml-results/{prefix}/{document_id}`).

        Unlike load(), raises on failure rather than returning None: a
        missing/unreachable source document should fail the task loudly (so
        it retries/dead-letters — see base_worker.py), not silently no-op.
        """
        resp = self._client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()
