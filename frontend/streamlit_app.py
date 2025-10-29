import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Medical Inference UI", layout="wide")
API_URL = "http://localhost:8000/predict"

st.title("🩺 AiCenna Diagnosis Platform")
st.markdown("Voting-Based Classification System")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify & Run", type="primary"):
        with st.spinner("Processing with multiple models..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = requests.post(API_URL, files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                with col2:
                    # CLIP Classification
                    st.subheader("Image Classification")
                    if data.get("clip_label"):
                        st.success(f"**Image Type:** {data['clip_label'].upper()}")
                    else:
                        st.error("CLIP classification failed")
                    
                    st.divider()
                    
                    # Final Prediction
                    st.subheader("Final Diagnosis ")
                    if data.get("error") and not data.get("prediction_label"):
                        st.error(data["error"])
                    else:
                        if data.get("prediction_label"):
                            # Extract Yes/No from prediction_label
                            pred_text = data['prediction_label']
                            if pred_text.startswith("Yes"):
                                st.error(f"**{pred_text}**")
                            elif pred_text.startswith("No"):
                                st.success(f"**{pred_text}**")
                            else:
                                st.info(f"**{pred_text}**")
                            
                            st.metric("Average Confidence", data['prediction_confidence'])
                            st.metric("Models Used", f"{data.get('successful_models', 0)}")
                        else:
                            st.warning("Could not determine prediction result.")
                    
                    st.divider()
                    
                    # Individual Model Results
                    st.subheader("Individual Model Predictions")
                    if data.get("model_results"):
                        for idx, result in enumerate(data["model_results"], 1):
                            with st.expander(f"**Model {idx}: {result['model']}**", expanded=False):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Prediction:** {result['prediction']}")
                                with col_b:
                                    st.write(f"**Confidence:** {result['confidence']}")
                    else:
                        st.info("No individual model results available")
                    
                    # Show warnings if any
                    if data.get("error") and data.get("prediction_label"):
                        st.warning(f"{data['error']}")