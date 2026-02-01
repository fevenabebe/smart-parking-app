import streamlit as st
import cv2
import numpy as np
import joblib
import os

# =========================
# Load model
# =========================
@st.cache_resource  # cache to avoid reloading every run
def load_model():
    model_path = "parking_detection_model.pkl"
    
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Make sure parking_detection_model.pkl is in the repo.")
        st.stop()
    
    data = joblib.load(model_path)
    svm_model = data["svm_model"]
    bg_subtractor = data["bg_subtractor"]
    
    return svm_model, bg_subtractor

svm_model, bg_subtractor = load_model()

# =========================
# Helper functions
# =========================
def preprocess_frame(frame):
    """Resize and convert to HSV if needed"""
    frame_resized = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    return frame_resized, hsv

def detect_parking(frame):
    """
    Dummy function to simulate detection.
    Replace with your actual detection logic using svm_model & bg_subtractor
    """
    # Example: just apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    return fg_mask

# =========================
# Streamlit App
# =========================
st.title("Smart Parking System 🚗🅿️")

st.write("Upload a parking video or image to detect free/occupied spaces.")

input_type = st.radio("Select input type:", ["Image", "Video"])

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed_image, hsv = preprocess_frame(image)
        mask = detect_parking(processed_image)
        st.image(mask, channels="GRAY", caption="Parking Detection Output")

elif input_type == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile)
        
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, hsv = preprocess_frame(frame)
            mask = detect_parking(processed_frame)
            stframe.image(mask, channels="GRAY")
        
        cap.release()
        os.remove(tfile)

st.write("✅ Model loaded successfully. Ready to detect parking!")
