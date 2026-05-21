import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.title("🎯 Object Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    model = YOLO("yolov8n.pt")

    results = model(image_np)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Objects", use_column_width=True)