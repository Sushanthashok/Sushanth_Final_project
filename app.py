# ----------------------------------------------------------
# Solar Panel Defect Detection Streamlit App (Fixed)
# ----------------------------------------------------------

import numpy as np
from PIL import Image
import os
import streamlit as st
import tensorflow as tf

# --- Page setup (put this early) ---
st.set_page_config(page_title="Solar Panel Defect Detection", layout="centered")
st.title("🔆 Solar Panel Defect Detection")
st.write("Upload a solar panel image to detect possible defects.")

# --- Config: update path/labels if needed ---
MODEL_PATH = "solar_panel_model.h5"  # or "models/solar_panel_model.h5"
IMG_SIZE = (224, 224)                # (width, height) used in training
CLASS_NAMES = [
    "Bird-Drop",
    "Clean",
    "Dusty",
    "Electrical-Damage",
    "Physical-Damage",
    "Snow-Covered",
]

# --- Load model safely (cached) ---
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model = load_model(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# --- Helpers ---
def _softmax_safe(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)

def preprocess(pil_img: Image.Image, size=(224, 224)) -> np.ndarray:
    # ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(size)
    arr = np.asarray(pil_img).astype("float32") / 255.0  # normalize 0..1
    return arr  # HWC float32

def predict_defect(pil_img: Image.Image, model, img_size=(224, 224)):
    arr = preprocess(pil_img, img_size)        # HWC
    batch = np.expand_dims(arr, axis=0)        # NHWC
    preds = model.predict(batch, verbose=0)

    # Flatten to 1D
    preds = np.squeeze(preds)
    if preds.ndim != 1:
        raise ValueError(f"Unexpected model output shape: {preds.shape}")

    # If logits, turn into probabilities
    if not np.all((preds >= 0) & (preds <= 1)) or abs(preds.sum() - 1) > 1e-3:
        preds = _softmax_safe(preds)

    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100.0
    return pred_idx, confidence, preds

# --- UI: Upload & Predict ---
uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Predicting..."):
            pred_idx, confidence, probs = predict_defect(image, model, IMG_SIZE)

        # sanity check for class names
        if len(CLASS_NAMES) != len(probs):
            st.warning(
                f"Model returns {len(probs)} outputs but CLASS_NAMES has {len(CLASS_NAMES)}. "
                "Update CLASS_NAMES to match your training order."
            )

        label = CLASS_NAMES[pred_idx] if len(CLASS_NAMES) == len(probs) else f"class_{pred_idx}"
        st.subheader("🔮 Prediction")
        st.success(f"**{label}**  (confidence: {confidence:.2f}%)")

        # Show top-3
        st.subheader("📊 Top probabilities")
        top_k = min(3, len(probs))
        top_idx = np.argsort(probs)[::-1][:top_k]
        for i in top_idx:
            name = CLASS_NAMES[i] if len(CLASS_NAMES) == len(probs) else f"class_{i}"
            st.write(f"- {name}: **{probs[i]*100:.2f}%**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("Upload a solar panel image to start.")
