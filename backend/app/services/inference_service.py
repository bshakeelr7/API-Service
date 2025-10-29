import logging
import numpy as np
from PIL import Image
from collections import Counter
from ..services.clip_service import classify_image_pil
from ..services.model_loader import load_model_metadata_and_get_instance
from ..models import ModelMeta
from ..db import SessionLocal
from .label_mappings import LABEL_MAPPINGS


def get_all_models_by_image_type(image_type: str):
    """Get all models for a specific image type"""
    try:
        db = SessionLocal()
        models = db.query(ModelMeta).filter(ModelMeta.image_type == image_type).all()
        db.close()
        return models
    except Exception as e:
        logging.warning(f"DB unavailable: {e}")
        return []


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


def parse_timm_result(preds, image_type):
    """Extract predicted label and confidence from TIMM model output"""
    try:
        # TIMM models output logits, apply softmax
        import torch
        if isinstance(preds, torch.Tensor):
            probs = torch.nn.functional.softmax(preds, dim=1)
            probs = probs.cpu().numpy()[0]
        else:
            probs = preds[0]
        
        label_index = np.argmax(probs)
        confidence = float(probs[label_index]) * 100
        
        # Get class names from label mappings
        class_names = list(LABEL_MAPPINGS.get(image_type, {}).keys())
        if label_index < len(class_names):
            class_label = class_names[label_index]
            return class_label, confidence
        else:
            return None, confidence
    except Exception as e:
        logging.warning(f"TIMM parse failed: {e}")
        return None, None


def run_single_model_inference(image_pil: Image.Image, model_meta, predicted_label):
    """Run inference on a single model"""
    try:
        framework, model_obj, _ = load_model_metadata_and_get_instance(model_meta)
    except Exception as e:
        logging.error(f"Model loading failed for {model_meta.model_name}: {e}")
        return None, None, str(e)

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

        elif framework == "timm":
            import torch
            from torchvision import transforms
            
            # TIMM standard preprocessing
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image_pil).unsqueeze(0)
            
            # Move to same device as model
            device = next(model_obj.parameters()).device
            img_tensor = img_tensor.to(device)
            
            model_obj.eval()
            with torch.no_grad():
                preds = model_obj(img_tensor)
            
            class_label, confidence = parse_timm_result(preds, predicted_label)

        else:
            return None, None, f"Unsupported framework: {framework}"

        return class_label, confidence, None

    except Exception as e:
        logging.error(f"Inference failed for {model_meta.model_name}: {e}")
        return None, None, str(e)


def vote_on_predictions(predictions):
    """
    Perform voting on predictions from multiple models
    predictions: list of tuples (class_label, confidence, model_name)
    Returns: final_label, average_confidence, vote_details
    """
    if not predictions:
        return None, None, []
    
    # Extract labels for voting
    labels = [pred[0] for pred in predictions if pred[0] is not None]
    
    if not labels:
        return None, None, []
    
    # Count votes
    vote_counts = Counter(labels)
    final_label = vote_counts.most_common(1)[0][0]
    
    # Calculate average confidence for the winning label
    confidences = [pred[1] for pred in predictions if pred[0] == final_label and pred[1] is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Create vote details
    vote_details = []
    for pred in predictions:
        vote_details.append({
            "model": pred[2],
            "prediction": pred[0],
            "confidence": f"{pred[1]:.2f}%" if pred[1] else "N/A"
        })
    
    return final_label, avg_confidence, vote_details


def do_inference(image_pil: Image.Image, labels=None):
    # --- 1. Classify with CLIP ---
    try:
        predicted_label, _ = classify_image_pil(image_pil, labels=labels)
    except Exception as e:
        return {
            "clip_label": None,
            "prediction_label": None,
            "prediction_confidence": None,
            "model_results": [],
            "error": f"CLIP failure: {e}"
        }

    # --- 2. Get ALL models for this image type ---
    model_metas = get_all_models_by_image_type(predicted_label)
    
    if not model_metas:
        return {
            "clip_label": predicted_label,
            "prediction_label": None,
            "prediction_confidence": None,
            "model_results": [],
            "error": "No models configured for this image type."
        }

    logging.info(f"Found {len(model_metas)} models for image type: {predicted_label}")

    # --- 3. Run inference on ALL models ---
    predictions = []
    errors = []
    
    for model_meta in model_metas:
        logging.info(f"Running inference with model: {model_meta.model_name}")
        class_label, confidence, error = run_single_model_inference(image_pil, model_meta, predicted_label)
        
        if error:
            errors.append(f"{model_meta.model_name}: {error}")
        elif class_label:
            predictions.append((class_label, confidence, model_meta.model_name))
        else:
            errors.append(f"{model_meta.model_name}: Could not determine label")

    # --- 4. Perform voting ---
    if not predictions:
        return {
            "clip_label": predicted_label,
            "prediction_label": None,
            "prediction_confidence": None,
            "model_results": [],
            "error": f"All models failed. Errors: {'; '.join(errors)}"
        }

    final_label, avg_confidence, vote_details = vote_on_predictions(predictions)

    # --- 5. Map label to Yes/No ---
    label_mapping = LABEL_MAPPINGS.get(predicted_label, {})
    yes_no = label_mapping.get(final_label, "Unknown")

    return {
        "clip_label": predicted_label,
        "prediction_label": f"{yes_no} ({final_label.replace('_', ' ')})",
        "prediction_confidence": f"{avg_confidence:.2f}%",
        "model_results": vote_details,
        "total_models": len(model_metas),
        "successful_models": len(predictions),
        "error": None if not errors else f"Some models had issues: {'; '.join(errors)}"
    }


