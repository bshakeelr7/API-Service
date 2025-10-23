import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Inference UI", layout="wide")
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000/predict")

st.title("Medical Inference — Upload Image")
uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)
    if st.button("Classify & Run Model"):
        with st.spinner("Uploading to API..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = requests.post(API_URL, files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                st.subheader("CLIP classification")
                st.write("Label:", data.get("clip_label"))
                st.write("Scores:", data.get("clip_scores"))
                st.subheader("Model inference")
                st.write("Model used:", data.get("model_used"))
                st.write("Framework:", data.get("model_framework"))
                st.write("Model path:", data.get("model_path"))
                if data.get("prediction"):
                    st.json(data.get("prediction"))
                if data.get("error"):
                    st.error(data.get("error"))
