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

DEFAULT_LABELS = ["brain", "chest", "skin", "derma"]

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
    best_idx = int(np.argmax(probs))
    scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return labels[best_idx], scores
