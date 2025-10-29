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

def load_timm_model(local_path, model_meta):
    """Load TIMM model with multiple fallback strategies"""
    import torch
    import timm
    from .label_mappings import LABEL_MAPPINGS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = len(LABEL_MAPPINGS.get(model_meta.image_type, {}))
    if num_classes == 0:
        logging.warning(f"No label mapping found for {model_meta.image_type}, using default 1000 classes")
        num_classes = 1000
    
    model_name = None
    file_name_lower = model_meta.file_name.lower()
    
    if "swin_tiny" in file_name_lower or "swin_tiny" in model_meta.model_name.lower():
        model_name = "swin_tiny_patch4_window7_224"
    elif "efficientnetv2_b0" in file_name_lower or "efficientnetv2_b0" in model_meta.model_name.lower():
        model_name = "tf_efficientnetv2_b0"
    elif "efficientnet" in file_name_lower:
        model_name = "efficientnet_b0"
    else:
        # Try to use the model_name directly
        model_name = model_meta.model_name
    
    logging.info(f"Attempting to load TIMM model: {model_name} with {num_classes} classes")
    
    # Strategy 1: Try loading as a complete saved model (torch.save(model, path))
    try:
        logging.info(f"Strategy 1: Loading complete model object")
        model_obj = torch.load(local_path, map_location=device)
        
        # If it's already a model object, use it directly
        if hasattr(model_obj, 'eval'):
            model_obj = model_obj.to(device)
            model_obj.eval()
            logging.info(f"Successfully loaded complete TIMM model object")
            return model_obj
    except Exception as e:
        logging.debug(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Try loading as state_dict and create model
    try:
        logging.info(f"Strategy 2: Loading state_dict and creating model")
        checkpoint = torch.load(local_path, map_location=device)
        
        # Create the model architecture
        model_obj = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # Extract state_dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Try to load state_dict
        model_obj.load_state_dict(state_dict, strict=True)
        model_obj = model_obj.to(device)
        model_obj.eval()
        logging.info(f"Successfully loaded TIMM model with state_dict (strict)")
        return model_obj
    except Exception as e:
        logging.debug(f"Strategy 2 (strict) failed: {e}")
    
    # Strategy 3: Try with strict=False to allow partial loading
    try:
        logging.info(f"Strategy 3: Loading state_dict with strict=False")
        checkpoint = torch.load(local_path, map_location=device)
        
        model_obj = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = model_obj.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logging.warning(f"Missing keys: {len(missing_keys)} keys")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
        
        model_obj = model_obj.to(device)
        model_obj.eval()
        logging.info(f"Successfully loaded TIMM model with state_dict (non-strict)")
        return model_obj
    except Exception as e:
        logging.debug(f"Strategy 3 failed: {e}")
    
    # Strategy 4: Load pretrained model and try to adapt classifier
    try:
        logging.info(f"Strategy 4: Loading pretrained model from timm hub")
        model_obj = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Try to load custom weights over pretrained
        try:
            checkpoint = torch.load(local_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            model_obj.load_state_dict(state_dict, strict=False)
            logging.info("Loaded custom weights over pretrained model")
        except:
            logging.warning("Could not load custom weights, using pretrained model")
        
        model_obj = model_obj.to(device)
        model_obj.eval()
        logging.info(f"Successfully loaded pretrained TIMM model")
        return model_obj
    except Exception as e:
        logging.debug(f"Strategy 4 failed: {e}")
    
    # Strategy 5: Just use pretrained model without custom weights
    try:
        logging.info(f"Strategy 5: Using pretrained model without custom weights")
        model_obj = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        model_obj = model_obj.to(device)
        model_obj.eval()
        logging.warning(f"Using pretrained {model_name} - could not load custom weights from {local_path}")
        return model_obj
    except Exception as e:
        logging.error(f"Strategy 5 failed: {e}")
        raise RuntimeError(f"All TIMM loading strategies failed for {model_name}")


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
        
        # --- YOLO/PyTorch Models ---
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
        
        # --- Keras/TensorFlow Models ---
        elif framework in ("keras", "tf", "h5"):
            try:
                import tensorflow as tf
                model_obj = tf.keras.models.load_model(local_path)
                MODEL_INSTANCE_CACHE[name] = ("keras", model_obj, local_path)
                return MODEL_INSTANCE_CACHE[name]
            except Exception as e:
                raise RuntimeError(f"Keras load failed: {e}")
        
        # --- TIMM Models ---
        elif framework == "timm":
            try:
                model_obj = load_timm_model(local_path, model_meta)
                MODEL_INSTANCE_CACHE[name] = ("timm", model_obj, local_path)
                return MODEL_INSTANCE_CACHE[name]
            except Exception as e:
                raise RuntimeError(f"TIMM load failed: {e}")
        
        else:
            raise RuntimeError(f"Unsupported framework: {framework}")