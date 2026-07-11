"""Download the FER2013 image dataset from Kaggle and arrange train/val/test splits."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "msambare_fer2013"
IMAGES_DIR = DATA_DIR / "images_resplit"

DATASET_ID = "msambare/fer2013"
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SEED = 1337


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FER2013 via kagglehub and build train/val/test folders. "
            "The Kaggle dataset provides train/test splits; this script creates a "
            "validation split by sampling from the training images."
        )
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of training images per class to allocate to validation (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used when shuffling training images before splitting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload the dataset and recreate the image folders from scratch.",
    )
    return parser.parse_args()


def _download_dataset() -> Path:
    print(f"[INFO] Downloading dataset '{DATASET_ID}' via kagglehub...")
    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"[INFO] Dataset cached at {dataset_path}")
    return dataset_path


def _mirror_raw_dataset(source_dir: Path, force: bool) -> Path:
    if RAW_DIR.exists():
        if force:
            print(f"[INFO] Removing existing raw directory at {RAW_DIR}")
            shutil.rmtree(RAW_DIR)
        else:
            print(f"[INFO] Raw directory already present at {RAW_DIR}, skipping copy.")
            return RAW_DIR

    print(f"[INFO] Copying dataset contents to {RAW_DIR}")
    shutil.copytree(source_dir, RAW_DIR)
    return RAW_DIR


def _split_train_val(
    class_files: List[Path],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    files = list(class_files)
    rng.shuffle(files)

    val_count = max(1, int(len(files) * val_ratio))
    val_files = files[:val_count]
    train_files = files[val_count:]
    return train_files, val_files


def _copy_images(files: List[Path], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, destination / src.name)
    return len(files)


def _prepare_splits(raw_dir: Path, val_ratio: float, seed: int, force: bool) -> Dict[str, Dict[str, int]]:
    if IMAGES_DIR.exists():
        if force:
            print(f"[INFO] Removing existing prepared images at {IMAGES_DIR}")
            shutil.rmtree(IMAGES_DIR)
        else:
            print(f"[INFO] Prepared dataset already exists at {IMAGES_DIR}, skipping recreation.")
            return {}

    train_src = raw_dir / "train"
    test_src = raw_dir / "test"
    if not train_src.exists() or not test_src.exists():
        raise FileNotFoundError(
            f"Unexpected dataset structure under {raw_dir}. "
            "Expected 'train/' and 'test/' folders as provided by the Kaggle dataset."
        )

    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    classes = sorted([p.name for p in train_src.iterdir() if p.is_dir()])
    for class_name in classes:
        class_train_files = sorted((train_src / class_name).glob("*"))
        train_files, val_files = _split_train_val(class_train_files, val_ratio, seed)

        summary["train"][class_name] += _copy_images(
            train_files, IMAGES_DIR / "train" / class_name
        )
        summary["val"][class_name] += _copy_images(
            val_files, IMAGES_DIR / "val" / class_name
        )

        test_files = sorted((test_src / class_name).glob("*"))
        summary["test"][class_name] += _copy_images(
            test_files, IMAGES_DIR / "test" / class_name
        )

    return summary


def _log_summary(summary: Dict[str, Dict[str, int]]) -> None:
    if not summary:
        return
    print("\n[INFO] Dataset summary:")
    total_images = 0
    for split in ("train", "val", "test"):
        split_counts = summary.get(split, {})
        split_total = sum(split_counts.values())
        total_images += split_total
        print(f"  {split:<5} -> {split_total} images")
        for class_name, count in sorted(split_counts.items()):
            print(f"     - {class_name:<8}: {count}")
    print(f"[INFO] Total images prepared: {total_images}")
    if total_images < 1000:
        raise RuntimeError(
            f"Expected at least 1000 images after preparation, but only found {total_images}."
        )


def main() -> None:
    args = parse_arguments()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_cache_dir = _download_dataset()
    raw_dir = _mirror_raw_dataset(dataset_cache_dir, force=args.force)
    summary = _prepare_splits(
        raw_dir=raw_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force,
    )
    _log_summary(summary)
    print(f"[INFO] FER2013 dataset ready at {IMAGES_DIR}")


if __name__ == "__main__":
    main()
