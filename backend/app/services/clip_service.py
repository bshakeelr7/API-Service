import logging
import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    logging.info(f"Loaded CLIP on {device}")
except Exception as e:
    logging.error(f"Failed to load CLIP: {e}")
    model = None
    preprocess = None


DEFAULT_LABELS = [
    "axial or sagittal brain MRI scan displaying brain tumor glioma meningioma pituitary adenoma with focal asymmetric abnormal mass lesion nodule or irregular contrast-enhanced bright ring structure indicating neoplasm",
    "posteroanterior or lateral chest X-ray radiograph displaying thoracic cavity with visible ribcage skeletal bones lung fields heart shadow and mediastinum structures in black and white contrast",
    "axial brain MRI scan showing perfectly symmetric bilateral butterfly-wing shaped lateral ventricles with homogeneous diffuse bright white matter signal and preserved gray-white matter differentiation characteristic of alzheimer dementia neurodegeneration with cortical preservation",
    "close-up dermatological clinical photograph showing skin surface with visible lesion pigmentation rash erythema melanoma basal cell carcinoma actinic keratosis dermatitis or epidermal discoloration abnormality",
    "photograph showing human person face body or portrait",
    "photograph showing domestic cat feline or kitten animal",
    "photograph showing brown chestnut nut or seed"
]


LABEL_MAPPING = {
    "axial or sagittal brain MRI scan displaying brain tumor glioma meningioma pituitary adenoma with focal asymmetric abnormal mass lesion nodule or irregular contrast-enhanced bright ring structure indicating neoplasm": "brain",
    "posteroanterior or lateral chest X-ray radiograph displaying thoracic cavity with visible ribcage skeletal bones lung fields heart shadow and mediastinum structures in black and white contrast": "chest",
    "axial brain MRI scan showing perfectly symmetric bilateral butterfly-wing shaped lateral ventricles with homogeneous diffuse bright white matter signal and preserved gray-white matter differentiation characteristic of alzheimer dementia neurodegeneration with cortical preservation": "alzheimer",
    "close-up dermatological clinical photograph showing skin surface with visible lesion pigmentation rash erythema melanoma basal cell carcinoma actinic keratosis dermatitis or epidermal discoloration abnormality": "skin",
    "photograph showing human person face body or portrait": "person",
    "photograph showing domestic cat feline or kitten animal": "cat",
    "photograph showing brown chestnut nut or seed": "chestnut"
}

def classify_image_pil(image_pil: Image.Image, labels=None):
    if model is None or preprocess is None:
        raise RuntimeError("CLIP model not loaded")
    if labels is None:
        labels = DEFAULT_LABELS
    
    image_input = preprocess(image_pil).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(labels).to(device)
    
    with torch.no_grad():
        logits_per_image, _ = model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    
    # Get top 3 predictions
    top3_idx = np.argsort(probs)[-3:][::-1]
    
    best_idx = top3_idx[0]
    predicted_label = labels[best_idx]
    original_category = LABEL_MAPPING.get(predicted_label, predicted_label)
    
    # Detailed logging for debugging
    logging.info("=" * 60)
    logging.info("CLIP Classification Results:")
    logging.info("=" * 60)
    for i, idx in enumerate(top3_idx, 1):
        cat = LABEL_MAPPING.get(labels[idx], "unknown")
        logging.info(f"  #{i}: {cat.upper()} - Confidence: {probs[idx]:.4f} ({probs[idx]*100:.2f}%)")
    logging.info("-" * 60)
    logging.info(f"FINAL PREDICTION: {original_category.upper()} (confidence: {probs[best_idx]:.4f})")
    logging.info("=" * 60)
    
    # Check for ambiguous predictions
    if len(top3_idx) >= 2:
        margin = probs[top3_idx[0]] - probs[top3_idx[1]]
        if margin < 0.10:
            second_cat = LABEL_MAPPING.get(labels[top3_idx[1]], "unknown")
            logging.warning(f"AMBIGUOUS: Small margin ({margin:.4f}) between {original_category} and {second_cat}")
    
    # Create scores with mapped labels
    scores = {LABEL_MAPPING.get(label, label): float(probs[i]) for i, label in enumerate(labels)}
    
    return original_category, scores