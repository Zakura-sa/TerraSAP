"""
Convert the Pavia University hyperspectral scene into ImageFolder patches for
TerraSAP FSCIL experiments. The script applies PCA to retain the top three
spectral components, extracts 64x64 patches centered on labeled pixels, and
upsamples them to 224x224 RGB PNGs. Session definitions follow the requested
5+4 class schedule (base session + four incremental sessions).

Usage:
    python scripts/prepare_paviau.py \\
        --src data/PaviaU \\
        --dst data/PaviaU_ImageFolder \\
        --train-ratio 0.65 \\
        --max-train-per-class 2000 \\
        --max-test-per-class 800
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.io import loadmat

CLASS_ID_TO_NAME = {
    1: "Asphalt",
    2: "Meadows",
    3: "Gravel",
    4: "Trees",
    5: "Painted metal sheets",
    6: "Bare soil",
    7: "Bitumen",
    8: "Self-blocking bricks",
    9: "Shadows",
}

SESSION_PLAN = [
    {"session_id": 0, "type": "base", "classes": ["Meadows", "Trees", "Bare soil", "Shadows", "Painted metal sheets"]},
    {"session_id": 1, "type": "incremental", "classes": ["Asphalt"]},
    {"session_id": 2, "type": "incremental", "classes": ["Bitumen"]},
    {"session_id": 3, "type": "incremental", "classes": ["Gravel"]},
    {"session_id": 4, "type": "incremental", "classes": ["Self-blocking bricks"]},
]

BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Pavia University dataset for FSCIL.")
    parser.add_argument("--src", type=pathlib.Path, default=pathlib.Path("data/PaviaU"), help="Directory with PaviaU.mat and PaviaU_gt.mat.")
    parser.add_argument("--dst", type=pathlib.Path, default=pathlib.Path("data/PaviaU_ImageFolder"), help="Output root in ImageFolder format.")
    parser.add_argument("--patch-size", type=int, default=64, help="Spatial size of cropped hyperspectral patches.")
    parser.add_argument("--upsample-size", type=int, default=224, help="Output resolution (ViT input).")
    parser.add_argument("--train-ratio", type=float, default=0.65, help="Fraction of samples per class routed to the training split.")
    parser.add_argument("--max-train-per-class", type=int, default=2000, help="Upper bound on train patches per class (0 = all).")
    parser.add_argument("--max-test-per-class", type=int, default=800, help="Upper bound on test patches per class (0 = all).")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for shuffling pixels.")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def compute_pca_rgb(cube: np.ndarray, components: int = 3) -> np.ndarray:
    h, w, c = cube.shape
    flat = cube.reshape(-1, c).astype(np.float32)
    flat = np.nan_to_num(flat)
    mean = np.mean(flat, axis=0, keepdims=True)
    flat -= mean
    cov = (flat.T @ flat) / max(flat.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:components]
    eigenvectors = eigvecs[:, order]
    projected = flat @ eigenvectors
    projected = projected.reshape(h, w, components)
    min_val = projected.min(axis=(0, 1), keepdims=True)
    max_val = projected.max(axis=(0, 1), keepdims=True)
    denom = np.clip(max_val - min_val, 1e-6, None)
    projected = (projected - min_val) / denom
    projected = np.clip(projected * 255.0, 0, 255).astype(np.uint8)
    return projected


def collect_coordinates(labels: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    coords = defaultdict(list)
    for r, c in zip(*np.nonzero(labels)):
        class_id = int(labels[r, c])
        if class_id == 0:
            continue
        coords[class_id].append((r, c))
    return coords


def split_coordinates(
    coords: Dict[int, List[Tuple[int, int]]],
    train_ratio: float,
    seed: int,
    max_train_per_class: int,
    max_test_per_class: int,
) -> Dict[str, Dict[int, List[Tuple[int, int]]]]:
    rng = np.random.default_rng(seed)
    result = {"train": {}, "test": {}}
    for class_id, class_coords in coords.items():
        shuffled = list(class_coords)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train_coords = shuffled[:split_idx]
        test_coords = shuffled[split_idx:]
        if max_train_per_class > 0:
            train_coords = train_coords[:max_train_per_class]
        if max_test_per_class > 0:
            test_coords = test_coords[:max_test_per_class]
        result["train"][class_id] = train_coords
        result["test"][class_id] = test_coords
    return result


def extract_patch(
    center: Tuple[int, int],
    pad_rgb: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    half = patch_size // 2
    r, c = center
    row = r + half
    col = c + half
    patch = pad_rgb[row - half : row + half, col - half : col + half]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        raise RuntimeError(f"Patch has wrong shape {patch.shape} for center {center}")
    return patch


def export_split(
    split_name: str,
    split_coords: Dict[int, List[Tuple[int, int]]],
    rgb_cube: np.ndarray,
    upsample_size: int,
    patch_size: int,
    dst_root: pathlib.Path,
) -> Dict[str, int]:
    dst_split = dst_root / split_name
    dst_split.mkdir(parents=True, exist_ok=True)
    pad_width = patch_size // 2
    padded = np.pad(rgb_cube, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode="reflect")
    stats = {}
    for class_id, coords in split_coords.items():
        class_name = CLASS_ID_TO_NAME[class_id]
        slug = sanitize_name(class_name)
        class_dir = dst_split / slug
        class_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for idx, (r, c) in enumerate(coords):
            patch = extract_patch((r, c), padded, patch_size)
            img = Image.fromarray(patch, mode="RGB").resize((upsample_size, upsample_size), BICUBIC)
            img.save(class_dir / f"{split_name}_{slug}_{r:03d}_{c:03d}_{idx:05d}.png")
            saved += 1
        stats[class_name] = saved
    return stats


def main() -> None:
    args = parse_args()
    cube_path = args.src / "PaviaU.mat"
    label_path = args.src / "PaviaU_gt.mat"
    if not cube_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing source files under {args.src}")

    cube = loadmat(cube_path.as_posix())["paviaU"].astype(np.float32)
    labels = loadmat(label_path.as_posix())["paviaU_gt"].astype(np.int32)
    rgb_cube = compute_pca_rgb(cube, components=3)
    coords = collect_coordinates(labels)
    splits = split_coordinates(coords, args.train_ratio, args.seed, args.max_train_per_class, args.max_test_per_class)

    if args.dst.exists():
        print(f"Destination {args.dst} already exists and will be extended.")
    args.dst.mkdir(parents=True, exist_ok=True)

    split_stats = {}
    for split in ["train", "test"]:
        split_stats[split] = export_split(split, splits[split], rgb_cube, args.upsample_size, args.patch_size, args.dst)

    metadata = {
        "dataset": "Pavia University",
        "source": {"cube": cube_path.as_posix(), "labels": label_path.as_posix()},
        "patch_size": args.patch_size,
        "upsample_size": args.upsample_size,
        "train_ratio": args.train_ratio,
        "max_train_per_class": args.max_train_per_class,
        "max_test_per_class": args.max_test_per_class,
        "class_map": CLASS_ID_TO_NAME,
        "session_plan": SESSION_PLAN,
        "stats": split_stats,
    }
    with open(args.dst / "metadata_paviau.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Done. Metadata saved to", args.dst / "metadata_paviau.json")


if __name__ == "__main__":
    main()
