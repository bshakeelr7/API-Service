import os
import logging
from minio import Minio
from ..config import MINIO_HOST, MINIO_PORT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE, MINIO_BUCKET

_minio_client = None

def get_minio_client():
    global _minio_client
    if _minio_client:
        return _minio_client
    try:
        endpoint = f"{MINIO_HOST}:{MINIO_PORT}"
        _minio_client = Minio(endpoint, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)
        try:
            if not _minio_client.bucket_exists(MINIO_BUCKET):
                logging.warning(f"Bucket {MINIO_BUCKET} does not exist.")
        except Exception:
            pass
        return _minio_client
    except Exception as e:
        logging.warning(f"MinIO unavailable: {e}")
        return None

def fetch_model_to_cache(minio_path: str, local_path: str) -> bool:
    if os.path.exists(local_path):
        return True
    client = get_minio_client()
    if client is None:
        logging.warning("No MinIO client; cannot fetch model.")
        return False
    try:
        client.fget_object(MINIO_BUCKET, minio_path, local_path)
        logging.info(f"Fetched {minio_path} -> {local_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to fetch from MinIO: {e}")
        return False
