import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ========== SET PARAMETERS ==========
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Load the model
MODEL_PATH = r"C:\Users\KEERTHI KRISHANA\Downloads\planenet_resnet101_final_model.h5"

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# ========== COLOR MAPS ==========
WALL_COLOR = np.array([0, 255, 0], dtype=np.uint8)  # Green for walls
FLOOR_COLOR = np.array([255, 0, 0], dtype=np.uint8)  # Red for floor

# ========== IoU METRIC ==========
def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 0

# ========== FUNCTION TO APPLY MASK OVERLAY ==========
def overlay_segmentation(original_img, mask):
    overlay = original_img.copy()
    overlay[mask == 0] = (0.5 * overlay[mask == 0] + 0.5 * WALL_COLOR).astype(np.uint8)
    overlay[mask == 1] = (0.5 * overlay[mask == 1] + 0.5 * FLOOR_COLOR).astype(np.uint8)
    return overlay

# ========== PREDICTION FUNCTION ==========
def predict_image(image, ground_truth_path=None):
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        input_img = np.expand_dims(img_resized, axis=0)

        # Model Prediction
        pred_mask = model.predict(input_img, verbose=0)[0]
        pred_mask = np.squeeze(pred_mask)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # Convert original image back to uint8
        img_uint8 = (img_resized * 255).astype(np.uint8)

        # Overlay segmentation
        overlayed_img = overlay_segmentation(img_uint8, pred_mask)

        # If ground truth is provided, compute metrics
        if ground_truth_path and os.path.exists(ground_truth_path):
            gt_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (IMG_WIDTH, IMG_HEIGHT))
            gt_mask = (gt_mask > 127).astype(np.uint8)

            acc = accuracy_score(gt_mask.flatten(), pred_mask.flatten())
            precision = precision_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=1)
            recall = recall_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=1)
            f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=1)
            iou = iou_score(gt_mask, pred_mask)

            return img_uint8, pred_mask, overlayed_img, (acc, precision, recall, f1, iou)
        
        return img_uint8, pred_mask, overlayed_img, None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

# ========== STREAMLIT UI ==========
st.title("Room Segmentation")

if model is None:
    st.error("Model failed to load. Please check the model file path.")
else:
    # Upload Image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    ground_truth_file = st.file_uploader("Upload ground truth mask (optional)", type=["png"])

    if uploaded_file:
        try:
            image = np.array(Image.open(uploaded_file))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                # Save ground truth temporarily
                gt_path = None
                if ground_truth_file:
                    gt_path = "temp_gt.png"
                    with open(gt_path, "wb") as f:
                        f.write(ground_truth_file.getbuffer())

                orig, mask, overlayed, metrics = predict_image(image, gt_path)

                if orig is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(orig, caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(overlayed, caption="Predicted Segmentation Overlay", use_column_width=True)

                    if metrics:
                        st.subheader("Performance Metrics")
                        st.metric("Accuracy", f"{metrics[0]:.4f}")
                        st.metric("Precision", f"{metrics[1]:.4f}")
                        st.metric("Recall", f"{metrics[2]:.4f}")
                        st.metric("F1-score", f"{metrics[3]:.4f}")
                        st.metric("IoU", f"{metrics[4]:.4f}")

                # Clean up temporary file
                if gt_path and os.path.exists(gt_path):
                    os.remove(gt_path)

        except Exception as e:
            st.error(f"Image processing error: {str(e)}")

    # Webcam capture
    if st.button("Capture from Webcam"):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam")
            else:
                ret, frame = cap.read()
                cap.release()

                if ret:
                    orig, mask, overlayed, _ = predict_image(frame)
                    if orig is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                    caption="Captured Image", 
                                    use_column_width=True)
                        with col2:
                            st.image(overlayed, 
                                    caption="Overlayed Result", 
                                    use_column_width=True)
                else:
                    st.error("Failed to capture image from webcam")
        except Exception as e:
            st.error(f"Webcam error: {str(e)}")
