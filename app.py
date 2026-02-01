import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os

@st.cache_resource
def load_model():
    data = joblib.load("parking_detection_model.pkl")
    return data["svm_model"], data["bg_subtractor"]

model, bg_sub = load_model()

st.title("🚗 Smart Parking Detection")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64, 64))

    foreground, mask = bg_sub.apply(img)
    features = bg_sub.extract_features(img, mask).reshape(1, -1)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]

    label = "Occupied 🚗" if pred == 1 else "Empty 🅿️"
    st.success(f"{label} ({prob*100:.2f}% confidence)")
