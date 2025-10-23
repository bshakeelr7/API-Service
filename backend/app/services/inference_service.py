import logging
from PIL import Image
from ..services.clip_service import classify_image_pil
from ..services.model_loader import load_model_metadata_and_get_instance
from ..models import ModelMeta
from ..db import SessionLocal

def get_model_meta_by_image_type(image_type: str):
    try:
        db = SessionLocal()
        meta = db.query(ModelMeta).filter(ModelMeta.image_type == image_type).first()
        db.close()
        return meta
    except Exception as e:
        logging.warning(f"DB unavailable: {e}")
        return None

def do_inference(image_pil: Image.Image, labels=None):
    try:
        predicted_label, clip_scores = classify_image_pil(image_pil, labels=labels)
    except Exception as e:
        return {"clip_label": None, "clip_scores": None, "error": f"CLIP failure: {e}"}

    model_meta = get_model_meta_by_image_type(predicted_label)
    if model_meta is None:
        return {
            "clip_label": predicted_label,
            "clip_scores": clip_scores,
            "model_used": None,
            "prediction": None,
            "error": "No model configured for this image type (DB down or no entry)."
        }

    try:
        framework, model_obj, local_path = load_model_metadata_and_get_instance(model_meta)
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return {
            "clip_label": predicted_label,
            "clip_scores": clip_scores,
            "model_used": model_meta.model_name,
            "prediction": None,
            "error": f"Model loading failed: {e}"
        }

    try:
        if framework == "torch":
            # ultralytics YOLO handling
            out = None
            try:
                if hasattr(model_obj, "predict"):
                    import numpy as np
                    arr = np.array(image_pil)
                    res = model_obj.predict(source=arr, imgsz=640)
                    out = {"raw": str(res)}
                else:
                    res = model_obj(image_pil) if callable(model_obj) else None
                    out = {"raw": str(res)}
            except Exception as e:
                logging.warning(f"YOLO inference issue: {e}")
                out = {"raw": f"YOLO inference error: {e}"}
            prediction = out
        elif framework == "keras":
            import numpy as np
            arr = np.array(image_pil.resize((224,224))) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            arr = arr.astype("float32")
            arr = arr.reshape((1,)+arr.shape)
            preds = model_obj.predict(arr)
            prediction = {"predictions": preds.tolist()}
        else:
            prediction = {"error": "Unsupported framework"}
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return {
            "clip_label": predicted_label,
            "clip_scores": clip_scores,
            "model_used": model_meta.model_name,
            "prediction": None,
            "error": f"Inference failed: {e}"
        }

    return {
        "clip_label": predicted_label,
        "clip_scores": clip_scores,
        "model_used": model_meta.model_name,
        "model_framework": framework,
        "model_path": model_meta.minio_path,
        "prediction": prediction,
        "error": None
    }
