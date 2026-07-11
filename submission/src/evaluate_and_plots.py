"""Evaluate the trained model on the FER2013 test set and generate plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    create_plain_dataset,
    get_class_names,
    get_images_dir,
    load_history,
    plot_confusion_matrix,
    plot_training_curves,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the FER2013 model and generate visualizations."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="submission/saved_model",
        help="Path to the trained model (SavedModel directory or .keras file).",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="submission/history.npy",
        help="Path to the training history file.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="submission/plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used when evaluating datasets.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(get_images_dir()),
        help="Directory that contains the prepared train/val/test folders.",
    )
    return parser.parse_args()


def _is_saved_model_dir(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "saved_model.pb").exists()


def _load_saved_model_with_tfsm_layer(model_dir: Path) -> tf.keras.Model:
    try:
        from keras.layers import TFSMLayer  # type: ignore
    except ImportError:  # pragma: no cover - fallback for older TF builds
        try:
            from tensorflow.keras.layers import TFSMLayer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "TFSMLayer is not available to load TensorFlow SavedModel artifacts."
            ) from exc

    layer = TFSMLayer(str(model_dir), call_endpoint="serving_default")
    inputs = tf.keras.Input(shape=(48, 48, 1), name="input_layer")
    outputs: Any = layer(inputs)
    if isinstance(outputs, dict):
        outputs_dict = outputs
        outputs = outputs_dict.get("output_0")
        if outputs is None:
            outputs = next(iter(outputs_dict.values()))

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="saved_model_wrapper")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(model_path: Path | str) -> tf.keras.Model:
    path = Path(model_path)
    if _is_saved_model_dir(path):
        try:
            model = tf.keras.models.load_model(path)
        except (ValueError, IOError):
            model = _load_saved_model_with_tfsm_layer(path)
    else:
        model = tf.keras.models.load_model(path)

    if not getattr(model, "optimizer", None):
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


def main() -> None:
    args = parse_arguments()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(args.model_path)
    history_payload = load_history(Path(args.history_path))
    history = history_payload.get("history", history_payload)

    test_ds = create_plain_dataset(
        "test",
        image_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        seed=1337,
        shuffle=False,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[RESULT] Test accuracy: {test_acc * 100:.2f}% | loss: {test_loss:.4f}")

    # Recreate dataset for deterministic iteration when collecting predictions.
    pred_ds = create_plain_dataset(
        "test",
        image_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        seed=1337,
        shuffle=False,
    )

    y_true_batches = []
    for _, labels in pred_ds:
        y_true_batches.append(labels.numpy())
    y_true = np.concatenate(y_true_batches, axis=0)

    y_prob = model.predict(pred_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    class_names = get_class_names()
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    plot_confusion_matrix(cm, class_names, plots_dir / "confusion_matrix.png")
    print("[INFO] Classification report:")
    print(report)

    plot_training_curves(history, plots_dir)

    train_best = max(history.get("accuracy", [0]))
    test_goal = test_acc >= 0.85
    train_goal = train_best >= 0.85
    print(
        f"[CHECK] Train accuracy goal {'met' if train_goal else 'NOT met'} "
        f"({train_best * 100:.2f}% best)."
    )
    print(
        f"[CHECK] Test accuracy goal {'met' if test_goal else 'NOT met'} "
        f"({test_acc * 100:.2f}%)."
    )


if __name__ == "__main__":
    main()
