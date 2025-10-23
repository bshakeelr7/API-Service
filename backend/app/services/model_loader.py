import os
import threading
import logging
from ..config import CACHE_DIR
from .minio_service import fetch_model_to_cache

os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_INSTANCE_CACHE = {}
MODEL_LOCKS = {}

def _get_lock(name):
    if name not in MODEL_LOCKS:
        MODEL_LOCKS[name] = threading.Lock()
    return MODEL_LOCKS[name]

def model_local_path(file_name: str) -> str:
    return os.path.join(CACHE_DIR, file_name)

def load_model_metadata_and_get_instance(model_meta):
    name = model_meta.model_name
    if name in MODEL_INSTANCE_CACHE:
        return MODEL_INSTANCE_CACHE[name]

    lock = _get_lock(name)
    with lock:
        if name in MODEL_INSTANCE_CACHE:
            return MODEL_INSTANCE_CACHE[name]

        local_path = model_local_path(model_meta.file_name)
        if not os.path.exists(local_path):
            ok = fetch_model_to_cache(model_meta.minio_path, local_path)
            if not ok:
                raise RuntimeError("Model file not available locally and MinIO unavailable.")

        framework = (model_meta.framework or "").lower()
        logging.info(f"Loading {name} from {local_path} as {framework}")
        if framework in ("yolo", "torch", "pytorch"):
            try:
                from ultralytics import YOLO
                model_obj = YOLO(local_path)
                MODEL_INSTANCE_CACHE[name] = ("torch", model_obj, local_path)
                return MODEL_INSTANCE_CACHE[name]
            except Exception as e:
                logging.warning(f"Ultralytics load failed: {e}. Trying torch.load")
                import torch
                model_obj = torch.load(local_path, map_location="cpu")
                MODEL_INSTANCE_CACHE[name] = ("torch", model_obj, local_path)
                return MODEL_INSTANCE_CACHE[name]
        elif framework in ("keras", "tf", "h5"):
            try:
                import tensorflow as tf
                model_obj = tf.keras.models.load_model(local_path)
                MODEL_INSTANCE_CACHE[name] = ("keras", model_obj, local_path)
                return MODEL_INSTANCE_CACHE[name]
            except Exception as e:
                raise RuntimeError(f"Keras load failed: {e}")
        else:
            raise RuntimeError(f"Unsupported framework: {framework}")
