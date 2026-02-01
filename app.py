import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Smart Parking System", layout="centered")

st.title("🚗 Smart Parking Spot Detection")
st.write("Upload a parking spot image to classify it as **Empty** or **Occupied**.")

# =========================
# Load model correctly
# =========================
@st.cache_resource
def load_model():
    data = joblib.load("parking_detection_model.pkl")
    svm_model = data["svm_model"]
    bg_subtractor = data["bg_subtractor"]
    return svm_model, bg_subtractor

try:
    svm_model, bg_subtractor = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model.")
    st.exception(e)
    st.stop()

# =========================
# Feature extraction
# =========================
def extract_features(image):
    foreground, mask = bg_subtractor.apply(image)
    features = bg_subtractor.extract_features(image, mask)
    return features, foreground, mask

# =========================
# UI
# =========================
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(image_bgr, (64, 64))

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Parking Spot"):
        try:
            features, foreground, mask = extract_features(resized)
            features = features.reshape(1, -1)

            pred = svm_model.predict(features)[0]
            probs = svm_model.predict_proba(features)[0]

            label = "Occupied 🚫" if pred == 1 else "Empty ✅"
            confidence = probs[pred] * 100

            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: **{confidence:.2f}%**")

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB), caption="Foreground")
            with col2:
                st.image(mask, caption="Mask", clamp=True)

        except Exception as e:
            st.error("Prediction failed")
            st.exception(e)
