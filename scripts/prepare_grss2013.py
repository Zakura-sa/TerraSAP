"""
Convert the Houston 2013 (IEEE GRSS Data Fusion) hyperspectral cube to
ImageFolder-style PNGs for TerraSAP FSCIL experiments. The script applies
3-component PCA, extracts 64x64 patches centered on labeled pixels (train/test
ROIs), upsamples them to 224x224 RGB, and saves files per class. Session split
follows the requested 7-class base session plus four incremental sessions.

Usage:
    python scripts/prepare_grss2013.py \\
        --src data/GRSS2013 \\
        --dst data/GRSS2013_ImageFolder \\
        --patch-size 64 \\
        --max-train-per-class 4000 \\
        --max-test-per-class 1500
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
from tifffile import imread

CLASS_ID_TO_NAME = {
    1: "Grass healthy",
    2: "Grass stressed",
    3: "Grass synthetic",
    4: "Tree",
    5: "Soil",
    6: "Water",
    7: "Residential",
    8: "Commercial",
    9: "Road",
    10: "Highway",
    11: "Railway",
    12: "Parking lot 1",
    13: "Parking lot 2",
    14: "Tennis court",
    15: "Running track",
}

SESSION_PLAN = [
    {
        "session_id": 0,
        "type": "base",
        "classes": [
            "Grass healthy",
            "Grass stressed",
            "Grass synthetic",
            "Tree",
            "Soil",
            "Water",
            "Running track",
        ],
    },
    {"session_id": 1, "type": "incremental", "classes": ["Residential", "Commercial"]},
    {"session_id": 2, "type": "incremental", "classes": ["Road", "Highway"]},
    {"session_id": 3, "type": "incremental", "classes": ["Railway", "Parking lot 1"]},
    {"session_id": 4, "type": "incremental", "classes": ["Tennis court", "Parking lot 2"]},
]

BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Houston 2013 dataset for FSCIL.")
    parser.add_argument("--src", type=pathlib.Path, default=pathlib.Path("data/GRSS2013"), help="Directory with CASI cube and ROI masks.")
    parser.add_argument("--dst", type=pathlib.Path, default=pathlib.Path("data/GRSS2013_ImageFolder"), help="Destination ImageFolder root.")
    parser.add_argument("--patch-size", type=int, default=64, help="Spatial sampling window size.")
    parser.add_argument("--upsample-size", type=int, default=224, help="Output resolution for ViT input.")
    parser.add_argument("--max-train-per-class", type=int, default=4000, help="Upper bound on train patches per class (0 = all).")
    parser.add_argument("--max-test-per-class", type=int, default=1500, help="Upper bound on test patches per class (0 = all).")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for shuffling coordinates.")
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


def collect_coordinates(mask: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    coords = defaultdict(list)
    for r, c in zip(*np.nonzero(mask)):
        class_id = int(mask[r, c])
        if class_id == 0:
            continue
        coords[class_id].append((r, c))
    return coords


def shuffle_and_limit(
    coords: Dict[int, List[Tuple[int, int]]],
    limit_per_class: int,
    rng: np.random.Generator,
) -> Dict[int, List[Tuple[int, int]]]:
    result = {}
    for class_id, points in coords.items():
        shuffled = list(points)
        rng.shuffle(shuffled)
        if limit_per_class > 0:
            shuffled = shuffled[:limit_per_class]
        result[class_id] = shuffled
    return result


def extract_patches(
    coords: Dict[int, List[Tuple[int, int]]],
    padded_rgb: np.ndarray,
    patch_size: int,
    split_name: str,
    dst_root: pathlib.Path,
    upsample_size: int,
) -> Dict[str, int]:
    dst_split = dst_root / split_name
    dst_split.mkdir(parents=True, exist_ok=True)
    stats = {}
    half = patch_size // 2
    for class_id, points in coords.items():
        class_name = CLASS_ID_TO_NAME[class_id]
        slug = sanitize_name(class_name)
        class_dir = dst_split / slug
        class_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for idx, (r, c) in enumerate(points):
            row = r + half
            col = c + half
            patch = padded_rgb[row - half : row + half, col - half : col + half]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                raise RuntimeError(f"Patch has wrong shape {patch.shape} for center {(r, c)}")
            img = Image.fromarray(patch, mode="RGB").resize((upsample_size, upsample_size), BICUBIC)
            img.save(class_dir / f"{split_name}_{slug}_{r:03d}_{c:03d}_{idx:05d}.png")
            saved += 1
        stats[class_name] = saved
    return stats


def main() -> None:
    args = parse_args()
    cube_path = args.src / "2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat"
    train_mask_path = args.src / "train_roi.tif"
    test_mask_path = args.src / "val_roi.tif"
    if not cube_path.exists():
        raise FileNotFoundError(f"Missing cube file: {cube_path}")
    for mask_path in [train_mask_path, test_mask_path]:
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing ROI mask: {mask_path}")

    cube = loadmat(cube_path.as_posix())["ans"].astype(np.float32)
    train_mask = imread(train_mask_path.as_posix()).astype(np.int32)
    test_mask = imread(test_mask_path.as_posix()).astype(np.int32)
    if cube.shape[:2] != train_mask.shape or cube.shape[:2] != test_mask.shape:
        raise RuntimeError("Cube and ROI masks have inconsistent spatial shapes.")

    rgb_cube = compute_pca_rgb(cube, components=3)
    pad_width = args.patch_size // 2
    padded = np.pad(rgb_cube, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode="reflect")

    rng = np.random.default_rng(args.seed)
    train_coords = shuffle_and_limit(collect_coordinates(train_mask), args.max_train_per_class, rng)
    test_coords = shuffle_and_limit(collect_coordinates(test_mask), args.max_test_per_class, rng)

    args.dst.mkdir(parents=True, exist_ok=True)
    stats = {
        "train": extract_patches(train_coords, padded, args.patch_size, "train", args.dst, args.upsample_size),
        "test": extract_patches(test_coords, padded, args.patch_size, "test", args.dst, args.upsample_size),
    }

    metadata = {
        "dataset": "Houston 2013 (GRSS Data Fusion)",
        "source": {
            "cube": cube_path.as_posix(),
            "train_roi": train_mask_path.as_posix(),
            "test_roi": test_mask_path.as_posix(),
        },
        "patch_size": args.patch_size,
        "upsample_size": args.upsample_size,
        "max_train_per_class": args.max_train_per_class,
        "max_test_per_class": args.max_test_per_class,
        "class_map": CLASS_ID_TO_NAME,
        "session_plan": SESSION_PLAN,
        "stats": stats,
    }
    with open(args.dst / "metadata_grss2013.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Conversion complete. Metadata written to", args.dst / "metadata_grss2013.json")


if __name__ == "__main__":
    main()
