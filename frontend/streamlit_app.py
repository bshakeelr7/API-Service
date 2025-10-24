
import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Medical Inference UI", layout="wide")
API_URL = "http://localhost:8000/predict"


st.title("🩺 AiCenna Diagnosis Platform")
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify & Run Model"):
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = requests.post(API_URL, files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                if data.get("error"):
                    st.error(data["error"])
                else:
                    st.subheader("CLIP Model Classification")
                    st.success(f"Classified as: {data.get('clip_label', 'Unknown')}")

                    st.subheader("Model Inference Result")
                    if data.get("prediction_label"):
                        st.info(f"Prediction: {data['prediction_label']}")
                        st.write(f"Confidence: {data['prediction_confidence']}")
                    else:
                        st.warning("Could not determine prediction result.")
