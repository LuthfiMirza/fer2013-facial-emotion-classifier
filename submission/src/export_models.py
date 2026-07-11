"""Export trained model to SavedModel, TFLite, and TFJS formats."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import tensorflow as tf
import tensorflowjs as tfjs

from utils import get_class_names


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained FER2013 model.")
    parser.add_argument(
        "--model-path",
        "--keras-model",
        dest="model_path",
        type=str,
        default="submission/saved_model",
        help="Path to the trained model (SavedModel directory or .keras file).",
    )
    parser.add_argument(
        "--saved-model-dir",
        type=str,
        default="submission/saved_model",
        help="Directory to store/refresh the TensorFlow SavedModel.",
    )
    parser.add_argument(
        "--tflite-dir",
        type=str,
        default="submission/tflite",
        help="Output directory for the TFLite model.",
    )
    parser.add_argument(
        "--tfjs-dir",
        type=str,
        default="submission/tfjs_model",
        help="Output directory for the TFJS model.",
    )
    return parser.parse_args()


def export_saved_model(model: tf.keras.Model, export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    model.export(export_dir)
    saved_model_path = export_dir / "saved_model.pb"
    if not saved_model_path.exists():
        raise FileNotFoundError(f"Failed to create SavedModel at {saved_model_path}")
    print(f"[INFO] SavedModel available at {saved_model_path}")
    return saved_model_path


def export_tflite(saved_model_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = output_dir / "model.tflite"
    tflite_path.write_bytes(tflite_model)

    labels_path = output_dir / "label.txt"
    labels_path.write_text("\n".join(get_class_names()))

    print(f"[INFO] TFLite model saved to {tflite_path}")
    print(f"[INFO] Labels saved to {labels_path}")
    return tflite_path


def export_tfjs(saved_model_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tfjs.converters.convert_tf_saved_model(str(saved_model_dir), str(output_dir))
    print(f"[INFO] TFJS model exported to {output_dir}")


def report_file_size(path: Path) -> None:
    size_mb = path.stat().st_size / (1024**2)
    print(f"   - {path.name}: {size_mb:.2f} MB")


def main() -> None:
    args = parse_arguments()

    model_path = Path(args.model_path)
    saved_model_dir = Path(args.saved_model_dir)
    tflite_dir = Path(args.tflite_dir)
    tfjs_dir = Path(args.tfjs_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path {model_path} not found. Train the model first."
        )

    source_is_saved_model = model_path.is_dir() and (model_path / "saved_model.pb").exists()

    if source_is_saved_model:
        if model_path.resolve() != saved_model_dir.resolve():
            if saved_model_dir.exists():
                shutil.rmtree(saved_model_dir)
            shutil.copytree(model_path, saved_model_dir, dirs_exist_ok=True)
        saved_model_source = saved_model_dir
    else:
        model = tf.keras.models.load_model(model_path)
        export_saved_model(model, saved_model_dir)
        saved_model_source = saved_model_dir

    tflite_path = export_tflite(saved_model_source, tflite_dir)
    export_tfjs(saved_model_source, tfjs_dir)

    print("[INFO] File size overview:")
    report_file_size(saved_model_dir / "saved_model.pb")
    report_file_size(tflite_path)

    tfjs_files = list(tfjs_dir.glob("*"))
    for file_path in tfjs_files:
        if file_path.is_file():
            report_file_size(file_path)


if __name__ == "__main__":
    main()
