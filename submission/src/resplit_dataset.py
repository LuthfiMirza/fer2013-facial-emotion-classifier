"""Utility script to reshuffle the prepared image dataset into new splits.

This is handy when working with a limited subset of FER2013 (or any dataset
already converted to per-class folders) and you want to reallocate the images
into a fresh train/validation/test split with deterministic shuffling.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR_DEFAULT = DATA_DIR / "images"
DEST_DIR_DEFAULT = DATA_DIR / "images_resplit"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resplit an existing image dataset organised as data/<split>/<class>/*. "
            "By default, the current train/val/test folders are merged, shuffled per "
            "class, and copied into a new directory with a 70/15/15 allocation."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DIR_DEFAULT,
        help=f"Root directory of existing split folders (default: {SOURCE_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEST_DIR_DEFAULT,
        help=f"Destination root for the resplit dataset (default: {DEST_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of samples per class assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of samples per class assigned to the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used when shuffling file lists.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of images kept per class before splitting. "
            "Useful for working with a smaller subset while maintaining balance."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the destination directory before writing the new split.",
    )
    return parser.parse_args()


def _list_class_files(source_dir: Path) -> Dict[str, List[Path]]:
    files_by_class: Dict[str, List[Path]] = {}
    for split_dir in sorted(source_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            files_by_class.setdefault(class_name, [])
            files_by_class[class_name].extend(sorted(class_dir.glob("*")))
    if not files_by_class:
        raise FileNotFoundError(
            f"No class folders found under {source_dir}. "
            "Expected structure: <source>/<split>/<class>/<images>."
        )
    return files_by_class


def _split_files(
    files: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    limit: int | None = None,
) -> Tuple[List[Path], List[Path], List[Path]]:
    files = list(files)
    random.Random(seed).shuffle(files)

    if limit is not None and limit > 0:
        files = files[:limit]

    total = len(files)
    if total == 0:
        return [], [], []

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # Ensure no split becomes empty when there are very few samples.
    if not val_files and test_files:
        val_files.append(test_files.pop())
    if not test_files and val_files:
        test_files.append(val_files.pop())

    return train_files, val_files, test_files


def _prepare_destination(dest_dir: Path, force: bool) -> None:
    if dest_dir.exists():
        if force:
            shutil.rmtree(dest_dir)
        else:
            raise FileExistsError(
                f"Destination directory {dest_dir} already exists. "
                "Use --force to overwrite."
            )
    for split in ("train", "val", "test"):
        (dest_dir / split).mkdir(parents=True, exist_ok=True)


def _copy_files(files: List[Path], target_dir: Path, class_name: str) -> None:
    class_dir = target_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for idx, src_path in enumerate(files):
        dst_name = f"{class_name}_{idx:05d}{src_path.suffix}"
        dst_path = class_dir / dst_name
        shutil.copy2(src_path, dst_path)


def main() -> None:
    args = parse_arguments()

    source_dir = args.source
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    dest_dir = args.dest
    _prepare_destination(dest_dir, args.force)

    files_by_class = _list_class_files(source_dir)

    summary: Dict[str, Dict[str, int]] = {}
    for class_name, files in sorted(files_by_class.items()):
        train_files, val_files, test_files = _split_files(
            files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit_per_class,
        )

        used_total = len(train_files) + len(val_files) + len(test_files)

        summary[class_name] = {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": used_total,
        }

        _copy_files(train_files, dest_dir / "train", class_name)
        _copy_files(val_files, dest_dir / "val", class_name)
        _copy_files(test_files, dest_dir / "test", class_name)

    total_counts = {"train": 0, "val": 0, "test": 0, "total": 0}
    print("[INFO] Resplit summary per class:")
    for class_name, counts in summary.items():
        print(
            f"  {class_name:<8} -> "
            f"train: {counts['train']:3d}, "
            f"val: {counts['val']:3d}, "
            f"test: {counts['test']:3d}, "
            f"total: {counts['total']:3d}"
        )
        total_counts["train"] += counts["train"]
        total_counts["val"] += counts["val"]
        total_counts["test"] += counts["test"]
        total_counts["total"] += counts["total"]

    print(
        "\n[INFO] Global totals -> "
        f"train: {total_counts['train']}, "
        f"val: {total_counts['val']}, "
        f"test: {total_counts['test']}, "
        f"total: {total_counts['total']}"
    )
    print(f"[INFO] Resplit dataset created at {dest_dir}")


if __name__ == "__main__":
    main()
