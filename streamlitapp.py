import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ========== SET PARAMETERS ==========
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load the model
MODEL_PATH = r"C:\Users\KEERTHI KRISHANA\Downloads\planenet_best_model.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # Disable compilation during loading
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Recompile manually
    return model

model = load_model()

# ========== COLOR MAPS ==========
WALL_COLOR = [0, 255, 0]  # Green
FLOOR_COLOR = [0, 0, 255]  # Blue

# ========== IMAGE PREPROCESSING FUNCTION ==========
def preprocess_image(image):
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resize image
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# ========== FUNCTION TO APPLY MASK OVERLAY ==========
def apply_mask(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = WALL_COLOR  # Walls = Green
    colored_mask[mask == 0] = FLOOR_COLOR  # Floor = Blue
    overlayed = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
    return overlayed

# ========== FUNCTION TO MAKE PREDICTIONS ==========
def predict_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = preprocess_image(image_rgb)
    predictions = model.predict(input_image)
    pred_mask = (predictions[0, :, :, 0] > 0.5).astype(np.uint8)  # Generate segmentation mask
    overlayed_image = apply_mask(image_rgb, pred_mask)
    return image_rgb, pred_mask, overlayed_image

# ========== STREAMLIT UI ==========
st.title("Room Segmentation")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        orig, mask, overlayed = predict_image(image)
        st.image(overlayed, caption="Overlayed Segmentation", use_column_width=True)

# ========== LIVE WEBCAM CAPTURE ==========
if st.button("Capture from Webcam"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig, mask, overlayed = predict_image(frame_rgb)
        st.image(frame_rgb, caption="Captured Image", use_column_width=True)
        st.image(overlayed, caption="Overlayed Result", use_column_width=True)
    else:
        st.error("Failed to capture image. Please check your webcam.")

