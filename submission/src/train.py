"""Training script for FER2013 CNN classifier with transfer learning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from build_model import build_sequential_cnn
from utils import (
    create_datasets,
    create_plain_dataset,
    get_images_dir,
    save_history,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FER2013 CNN classifier.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Epochs to train with the pretrained backbone frozen.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=20,
        help="Additional epochs with the backbone (partially) unfrozen.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate used during the frozen stage.",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-4,
        help="Learning rate used during fine-tuning.",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=60,
        help=(
            "Number of layers (counted from the end) of the backbone to unfreeze "
            "during fine-tuning. Use 0 to unfreeze the entire backbone."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(get_images_dir()),
        help="Path to the prepared image dataset.",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Disable on-the-fly data augmentation for the training split.",
    )
    return parser.parse_args()


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile the model with a fresh Adam optimizer."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_callbacks(
    checkpoint_cb: tf.keras.callbacks.Callback,
    *,
    patience: int = 10,
) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks sharing the provided checkpoint callback."""
    return [
        checkpoint_cb,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            min_delta=0.005,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def get_backbone(model: tf.keras.Model) -> Optional[tf.keras.Model]:
    """Return the MobileNetV2 backbone if present; otherwise None."""
    try:
        return model.get_layer("mobilenet_v2_base")
    except ValueError:
        return None


def unfreeze_backbone(backbone: tf.keras.Model, trainable_layers: int) -> None:
    """Enable training for the last `trainable_layers` layers of the backbone."""
    if trainable_layers <= 0:
        for layer in backbone.layers:
            layer.trainable = True
        print("[INFO] Unfreezing the entire backbone.")
        return

    total_layers = len(backbone.layers)
    start_index = max(total_layers - trainable_layers, 0)
    for idx, layer in enumerate(backbone.layers):
        layer.trainable = idx >= start_index
    print(
        f"[INFO] Unfreezing backbone layers {start_index}..{total_layers - 1} "
        f"({total_layers - start_index} layers)."
    )


def main() -> None:
    args = parse_arguments()

    np.random.seed(1337)
    tf.keras.utils.set_random_seed(1337)

    image_dir = Path(args.data_dir)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory {image_dir} does not exist. "
            "Run download_and_prepare.py to fetch and prepare the dataset."
        )

    train_ds, val_ds, _, class_names = create_datasets(
        image_dir=image_dir,
        batch_size=args.batch_size,
        seed=1337,
        augment=not args.disable_augmentation,
    )

    model = build_sequential_cnn(input_shape=(48, 48, 1), num_classes=len(class_names))
    compile_model(model, args.lr)

    saved_model_dir = Path("submission/saved_model")
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = saved_model_dir / "best.keras"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    print("[INFO] Stage 1: training with backbone frozen.")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=build_callbacks(checkpoint_cb, patience=10),
    )

    total_epochs = len(history.history.get("accuracy", []))

    backbone = get_backbone(model)
    if args.fine_tune_epochs > 0 and backbone is not None:
        print("[INFO] Stage 2: fine-tuning the backbone.")
        unfreeze_backbone(backbone, args.unfreeze_layers)
        compile_model(model, args.fine_tune_lr)

        fine_history = model.fit(
            train_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=total_epochs,
            validation_data=val_ds,
            callbacks=build_callbacks(checkpoint_cb, patience=12),
        )

        for key, values in fine_history.history.items():
            history.history.setdefault(key, [])
            history.history[key].extend(values)
        history.epoch.extend(fine_history.epoch)
        total_epochs += len(fine_history.history.get("accuracy", []))

    history.params["epochs"] = total_epochs
    history_path = Path("submission/history.npy")
    save_history(history, history_path)

    best_model = tf.keras.models.load_model(checkpoint_path)

    train_eval_ds = create_plain_dataset(
        "train",
        image_dir=image_dir,
        batch_size=args.batch_size,
        seed=1337,
        shuffle=False,
    )
    val_eval_ds = create_plain_dataset(
        "val",
        image_dir=image_dir,
        batch_size=args.batch_size,
        seed=1337,
        shuffle=False,
    )

    train_loss, train_acc = best_model.evaluate(train_eval_ds, verbose=0)
    val_loss, val_acc = best_model.evaluate(val_eval_ds, verbose=0)
    print(f"[RESULT] Train accuracy: {train_acc * 100:.2f}% | loss: {train_loss:.4f}")
    print(f"[RESULT] Val accuracy:   {val_acc * 100:.2f}% | loss: {val_loss:.4f}")

    best_model.export(saved_model_dir)
    print(f"[INFO] Exported TensorFlow SavedModel to {saved_model_dir}")

    test_eval_ds = create_plain_dataset(
        "test",
        image_dir=image_dir,
        batch_size=args.batch_size,
        seed=1337,
        shuffle=False,
    )
    test_loss, test_acc = best_model.evaluate(test_eval_ds, verbose=0)
    print(f"[RESULT] Test accuracy:  {test_acc * 100:.2f}% | loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
