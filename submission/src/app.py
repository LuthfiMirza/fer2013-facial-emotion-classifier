"""Streamlit UI for uploading images or using the camera to detect facial emotions."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Ensure helper utilities are importable
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import (  # noqa: E402  pylint: disable=wrong-import-position
    DEFAULT_IMAGE_SIZE,
    prediction_to_metadata,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "saved_model" / "best.keras"
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


@st.cache_resource(show_spinner="Memuat model TensorFlow...")
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tidak menemukan model tersimpan di {MODEL_PATH}. "
            "Pastikan sudah menjalankan submission/src/train.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
    """Convert an image to the model's expected format."""
    grayscale = image.convert("L").resize(DEFAULT_IMAGE_SIZE)
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=(0, -1))  # shape (1, 48, 48, 1)
    return array, grayscale


def run_inference(image: Image.Image, model: tf.keras.Model) -> Tuple[str, float, np.ndarray]:
    """Run the model and return predicted label, probability, and full predictions."""
    input_tensor, grayscale = preprocess_image(image)
    probs = model.predict(input_tensor, verbose=0)[0]
    top_index = int(np.argmax(probs))
    label = CLASS_NAMES[top_index]
    confidence = float(probs[top_index])
    return label, confidence, probs


def display_prediction(label: str, confidence: float) -> None:
    metadata = prediction_to_metadata(CLASS_NAMES.index(label))
    st.markdown(
        f"""
        ### Prediksi: **{metadata['label'].title()}** {metadata['emoji']}
        - Keyakinan model: **{confidence * 100:.2f}%**
        - Playlist: [{metadata['playlist_url']}]({metadata['playlist_url']})
        """,
        unsafe_allow_html=True,
    )


def show_probabilities(probs: np.ndarray) -> None:
    top_indices = np.argsort(probs)[::-1][:5]
    top_labels = [CLASS_NAMES[i].title() for i in top_indices]
    top_probs = probs[top_indices]
    st.bar_chart({"Probabilitas": top_probs}, x=top_labels)


def main() -> None:
    st.set_page_config(page_title="Deteksi Emosi Wajah", page_icon="😊", layout="wide")
    st.title("Deteksi Emosi Wajah FER2013")
    st.write(
        "Unggah foto wajah atau ambil gambar dari kamera untuk memprediksi emosi "
        "menggunakan model TensorFlow yang sudah dilatih."
    )

    model = load_model()

    input_method = st.sidebar.radio(
        "Pilih sumber gambar",
        options=("Upload File", "Kamera (snapshot)"),
    )

    image: Image.Image | None = None
    if input_method == "Upload File":
        uploaded = st.file_uploader("Pilih gambar wajah", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.read()))
    else:
        camera_image = st.camera_input("Ambil foto wajah")
        if camera_image is not None:
            image = Image.open(camera_image)

    if image is None:
        st.info("Silakan unggah gambar atau ambil foto terlebih dahulu.")
        return

    col_preview, col_result = st.columns([1, 1])
    with col_preview:
        st.image(image, caption="Input", use_column_width=True)

    label, confidence, probs = run_inference(image, model)
    with col_result:
        display_prediction(label, confidence)
        show_probabilities(probs)


if __name__ == "__main__":
    main()
