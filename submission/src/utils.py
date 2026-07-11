"""Utility helpers for the FER2013 facial emotion classification project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Dataset-wide constants
CLASS_NAMES: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EMOJI_MAP: Dict[str, str] = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

PLAYLIST_MAP: Dict[str, str] = {
    "angry": "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0",
    "disgust": "https://open.spotify.com/playlist/37i9dQZF1DWSY7w8O7aB6b",
    "fear": "https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP",
    "happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
    "sad": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1",
    "surprise": "https://open.spotify.com/playlist/37i9dQZF1DWUoqEG4xt4V1",
    "neutral": "https://open.spotify.com/playlist/37i9dQZF1DXbITWG1ZJKYt",
}

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_IMAGE_SIZE = (48, 48)


def get_project_root() -> Path:
    """Return the root directory of the submission project."""
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Return the base data directory."""
    return get_project_root() / "data"


def get_images_dir() -> Path:
    """Return the directory that holds extracted FER2013 images."""
    data_dir = get_data_dir()
    preferred = data_dir / "images_resplit"
    if preferred.exists():
        return preferred
    return data_dir / "images"


def get_class_names() -> List[str]:
    """Return ordered list of FER2013 label names."""
    return CLASS_NAMES.copy()


def get_data_augmentation_layer() -> tf.keras.Sequential:
    """Create the augmentation pipeline applied on-the-fly for training only."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=1337),
            tf.keras.layers.RandomRotation(0.08, seed=1337),
            tf.keras.layers.RandomZoom(0.1, seed=1337),
            tf.keras.layers.RandomContrast(0.1, seed=1337),
        ],
        name="data_augmentation",
    )


def _load_directory_split(
    split: str,
    image_dir: Path,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Load a dataset split from a directory structure."""
    split_dir = image_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Expected directory for split '{split}' at {split_dir}, but it does not exist."
        )

    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=DEFAULT_IMAGE_SIZE,
        shuffle=shuffle,
        seed=seed,
    )


def create_datasets(
    image_dir: Path | None = None,
    batch_size: int = 64,
    seed: int = 1337,
    augment: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Create train, validation, and test datasets from directory of images."""
    if image_dir is None:
        image_dir = get_images_dir()
    image_dir = Path(image_dir)

    class_names = get_class_names()

    train_ds = _load_directory_split("train", image_dir, batch_size, seed, shuffle=True)
    val_ds = _load_directory_split("val", image_dir, batch_size, seed, shuffle=False)
    test_ds = _load_directory_split("test", image_dir, batch_size, seed, shuffle=False)

    augmenter = get_data_augmentation_layer() if augment else None

    def prepare(
        dataset: tf.data.Dataset,
        training: bool,
    ) -> tf.data.Dataset:
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=AUTOTUNE,
        )
        if training and augment and augmenter is not None:
            dataset = dataset.map(
                lambda x, y: (augmenter(x, training=True), y),
                num_parallel_calls=AUTOTUNE,
            )
            return dataset.prefetch(AUTOTUNE)

        dataset = dataset.cache()
        return dataset.prefetch(AUTOTUNE)

    train_ds = prepare(train_ds, training=True)
    val_ds = prepare(val_ds, training=False)
    test_ds = prepare(test_ds, training=False)

    return train_ds, val_ds, test_ds, class_names


def create_plain_dataset(
    split: str,
    image_dir: Path | None = None,
    batch_size: int = 64,
    seed: int = 1337,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create a dataset split without data augmentation."""
    if image_dir is None:
        image_dir = get_images_dir()
    image_dir = Path(image_dir)

    dataset = _load_directory_split(split, image_dir, batch_size, seed, shuffle=shuffle)
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.cache().prefetch(AUTOTUNE)


def plot_training_curves(history: Dict[str, Iterable[float]], output_dir: Path) -> None:
    """Plot accuracy and loss curves and save them into the plots directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    acc_path = output_dir / "acc.png"
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    loss_path = output_dir / "loss.png"
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    print(f"[INFO] Saved accuracy plot to {acc_path}")
    print(f"[INFO] Saved loss plot to {loss_path}")


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    output_path: Path,
) -> None:
    """Save a confusion matrix figure to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix plot to {output_path}")


def label_from_index(index: int) -> str:
    """Translate numeric label index to class name."""
    return CLASS_NAMES[int(index)]


def prediction_to_metadata(index: int) -> Dict[str, str]:
    """Return metadata (class, emoji, playlist) for a predicted index."""
    label = label_from_index(index)
    return {
        "label": label,
        "emoji": EMOJI_MAP.get(label, ""),
        "playlist_url": PLAYLIST_MAP.get(label, ""),
    }


def load_image_for_inference(
    image_path: Path,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Load and preprocess a single image for inference."""
    from PIL import Image  # Imported lazily to avoid unnecessary dependency cost.

    image = Image.open(image_path).convert("L")
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=-1)
    return array


def save_history(history: tf.keras.callbacks.History, path: Path) -> None:
    """Persist training history for later reuse."""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": history.params,
        "epoch": history.epoch,
        "history": history.history,
    }
    if history_path.suffix == ".npy":
        np.save(history_path, payload, allow_pickle=True)
    else:
        history_path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Saved training history to {history_path}")


def load_history(path: Path) -> Dict:
    """Load history JSON produced by save_history."""
    history_path = Path(path)
    if not history_path.exists():
        raise FileNotFoundError(f"Cannot find history file at {history_path}")
    if history_path.suffix == ".npy":
        payload = np.load(history_path, allow_pickle=True).item()
        return payload
    return json.loads(history_path.read_text())
