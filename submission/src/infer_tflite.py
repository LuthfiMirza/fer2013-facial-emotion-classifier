"""Run inference using the exported TFLite model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils import load_image_for_inference, prediction_to_metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TFLite inference for FER2013 images.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="submission/tflite/model.tflite",
        help="Path to the TFLite model file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a 48x48 grayscale image for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model {model_path} not found.")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file {image_path} not found.")

    image_data = load_image_for_inference(image_path)
    input_data = np.expand_dims(image_data, axis=0).astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_details["index"])[0]

    top_index = int(np.argmax(probabilities))
    top_prob = float(probabilities[top_index])
    metadata = prediction_to_metadata(top_index)

    print(f"[INFO] Inference on image: {image_path}")
    print(f"[PRED] Label: {metadata['label']} | prob: {top_prob:.4f}")
    print(f"[PRED] Emoji: {metadata['emoji']}")
    print(f"[PRED] Playlist: {metadata['playlist_url']}")

    top3_indices = np.argsort(probabilities)[::-1][:3]
    print("[PRED] Top-3 distribution:")
    for idx in top3_indices:
        info = prediction_to_metadata(int(idx))
        print(f"   - {info['label']:<8} -> {probabilities[idx]:.4f}")


if __name__ == "__main__":
    main()
