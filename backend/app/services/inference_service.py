import logging
import numpy as np
from PIL import Image
from ..services.clip_service import classify_image_pil
from ..services.model_loader import load_model_metadata_and_get_instance
from ..models import ModelMeta
from ..db import SessionLocal
from .label_mappings import LABEL_MAPPINGS


def get_model_meta_by_image_type(image_type: str):
    try:
        db = SessionLocal()
        meta = db.query(ModelMeta).filter(ModelMeta.image_type == image_type).first()
        db.close()
        return meta
    except Exception as e:
        logging.warning(f"DB unavailable: {e}")
        return None


def parse_yolo_result(res):
    """Extract class label and confidence from YOLO (ultralytics) result"""
    try:
        if hasattr(res[0], "probs") and hasattr(res[0].probs, "top1conf"):
            probs = res[0].probs
            label_index = probs.top1
            confidence = float(probs.top1conf) * 100
            class_label = res[0].names[label_index]
            return class_label, confidence
        else:
            return None, None
    except Exception as e:
        logging.warning(f"YOLO parse failed: {e}")
        return None, None


def parse_keras_result(preds, image_type):
    """Extract predicted label and confidence from Keras model output"""
    try:
        preds = preds[0]
        label_index = np.argmax(preds)
        confidence = float(preds[label_index]) * 100
        class_names = list(LABEL_MAPPINGS.get(image_type, {}).keys())
        if label_index < len(class_names):
            class_label = class_names[label_index]
            return class_label, confidence
        else:
            return None, confidence
    except Exception as e:
        logging.warning(f"Keras parse failed: {e}")
        return None, None


def do_inference(image_pil: Image.Image, labels=None):
    # --- 1. Classify with CLIP ---
    try:
        predicted_label, _ = classify_image_pil(image_pil, labels=labels)
    except Exception as e:
        return {"clip_label": None, "prediction_label": None, "prediction_confidence": None, "error": f"CLIP failure: {e}"}

    # --- 2. Load model metadata ---
    model_meta = get_model_meta_by_image_type(predicted_label)
    if model_meta is None:
        return {
            "clip_label": predicted_label,
            "prediction_label": None,
            "prediction_confidence": None,
            "error": "No model configured for this image type."
        }

    # --- 3. Load model ---
    try:
        framework, model_obj, _ = load_model_metadata_and_get_instance(model_meta)
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return {
            "clip_label": predicted_label,
            "prediction_label": None,
            "prediction_confidence": None,
            "error": f"Model loading failed: {e}"
        }

    # --- 4. Run inference ---
    try:
        class_label, confidence = None, None

        if framework == "torch":
            # YOLO handling
            arr = np.array(image_pil)
            res = model_obj.predict(source=arr, imgsz=640)
            class_label, confidence = parse_yolo_result(res)

        elif framework == "keras":
            arr = np.array(image_pil.resize((224, 224))) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = arr.astype("float32").reshape((1,) + arr.shape)
            preds = model_obj.predict(arr)
            class_label, confidence = parse_keras_result(preds, predicted_label)

        else:
            return {"clip_label": predicted_label, "prediction_label": None, "prediction_confidence": None, "error": "Unsupported framework"}

        # --- 5. Map label to Yes/No ---
        if class_label:
            label_mapping = LABEL_MAPPINGS.get(predicted_label, {})
            yes_no = label_mapping.get(class_label, "Unknown")

            return {
                "clip_label": predicted_label,
                "prediction_label": f"{yes_no} ({class_label.replace('_', ' ')})",
                "prediction_confidence": f"{confidence:.2f}%",
                "error": None
            }
        else:
            return {
                "clip_label": predicted_label,
                "prediction_label": None,
                "prediction_confidence": None,
                "error": "Could not determine label from model output."
            }

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return {
            "clip_label": predicted_label,
            "prediction_label": None,
            "prediction_confidence": None,
            "error": f"Inference failed: {e}"
        }
