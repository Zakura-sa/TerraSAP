"""
Convert the So2Sat-LCZ42 HDF5 files to ImageFolder-format PNGs suitable for
FSCIL experiments. The conversion follows the Base7 + Inc10 session plan and
upsamples each 32x32 Sentinel-2 patch to 224x224 RGB (B4/B3/B2).

Usage example:
    python scripts/prepare_so2sat.py \
        --src data/So2Sat-LCZ42 \
        --dst data/So2Sat-LCZ42_ImageFolder \
        --train-limit 2500 \
        --test-limit 500
"""

import argparse
import pathlib
from collections import defaultdict

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

ALL_CLASSES = [
    "LCZ1_compact_highrise",
    "LCZ2_compact_midrise",
    "LCZ3_compact_lowrise",
    "LCZ4_open_highrise",
    "LCZ5_open_midrise",
    "LCZ6_open_lowrise",
    "LCZ7_lightweight_lowrise",
    "LCZ8_large_lowrise",
    "LCZ9_sparsely_built",
    "LCZ10_heavy_industry",
    "LCZA_dense_trees",
    "LCZB_scattered_trees",
    "LCZC_bush_scrub",
    "LCZD_low_plants",
    "LCZE_bare_rock_paved",
    "LCZF_bare_soil_sand",
    "LCZG_water",
]

# Desired class order for FSCIL (Base7 + 5 increments × 2 classes)
CLASS_ORDER = [
    "LCZA_dense_trees",       # Base
    "LCZB_scattered_trees",
    "LCZD_low_plants",
    "LCZG_water",
    "LCZ1_compact_highrise",
    "LCZ4_open_highrise",
    "LCZ8_large_lowrise",
    "LCZ2_compact_midrise",   # Session 1
    "LCZ5_open_midrise",
    "LCZ3_compact_lowrise",   # Session 2
    "LCZ6_open_lowrise",
    "LCZ9_sparsely_built",    # Session 3
    "LCZ10_heavy_industry",
    "LCZC_bush_scrub",        # Session 4
    "LCZE_bare_rock_paved",
    "LCZF_bare_soil_sand",    # Session 5
    "LCZ7_lightweight_lowrise",
]

SPLIT_FILES = {
    "train": ["training.h5", "validation.h5"],
    "test": ["testing.h5"],
}

BICUBIC = (
    Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare So2Sat-LCZ42 FSCIL dataset.")
    parser.add_argument(
        "--src",
        type=pathlib.Path,
        default=pathlib.Path("data/So2Sat-LCZ42"),
        help="Directory containing *_h5 files.",
    )
    parser.add_argument(
        "--dst",
        type=pathlib.Path,
        default=pathlib.Path("data/So2Sat-LCZ42_ImageFolder"),
        help="Target ImageFolder root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Number of patches to load per iteration.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=2500,
        help="Maximum samples per class for the training split (0 = no limit).",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=500,
        help="Maximum samples per class for the test split (0 = no limit).",
    )
    return parser.parse_args()


def normalize_tile(tile: np.ndarray) -> np.ndarray:
    """
    Convert Sentinel-2 B4/B3/B2 channels to uint8 RGB.
    Values are clipped to [0, 10000] then contrast-stretched to [0, 255].
    """
    tile = np.nan_to_num(tile).astype(np.float32)
    tile = np.clip(tile, 0.0, 10000.0)
    min_val = tile.min(axis=(0, 1), keepdims=True)
    max_val = tile.max(axis=(0, 1), keepdims=True)
    denom = np.clip(max_val - min_val, 1.0, None)
    tile = (tile - min_val) / denom
    tile = np.clip(tile * 255.0, 0, 255).astype(np.uint8)
    return tile


def convert_split(
    split: str,
    files: list[str],
    src_dir: pathlib.Path,
    dst_dir: pathlib.Path,
    batch_size: int,
    limit_per_class: int,
):
    counters = defaultdict(int)
    total_written = 0
    (dst_dir / split).mkdir(parents=True, exist_ok=True)
    for class_name in ALL_CLASSES:
        (dst_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for filename in files:
        h5_path = src_dir / filename
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing file: {h5_path}")

        print(f"[{split}] reading {h5_path}")
        with h5py.File(h5_path, "r") as f:
            sen2 = f["sen2"]
            labels = f["label"]
            total = sen2.shape[0]
            print(f"[{split}] patches in {filename}: {total}")

            for start in tqdm(range(0, total, batch_size), desc=f"{split}:{filename}", unit="patch"):
                end = min(start + batch_size, total)
                batch_sen2 = sen2[start:end]
                batch_label = labels[start:end]

                for local_idx in range(end - start):
                    label_vec = batch_label[local_idx]
                    class_idx = int(np.argmax(label_vec))
                    class_name = ALL_CLASSES[class_idx]

                    if limit_per_class > 0 and counters[class_name] >= limit_per_class:
                        continue

                    rgb = np.stack(
                        [
                            batch_sen2[local_idx, :, :, 2],  # B4 (red)
                            batch_sen2[local_idx, :, :, 1],  # B3 (green)
                            batch_sen2[local_idx, :, :, 0],  # B2 (blue)
                        ],
                        axis=-1,
                    )
                    rgb = normalize_tile(rgb)
                    img = Image.fromarray(rgb, mode="RGB").resize((224, 224), BICUBIC)

                    out_dir = dst_dir / split / class_name
                    img.save(out_dir / f"{split}_{filename}_{start+local_idx:07d}.png")
                    counters[class_name] += 1
                    total_written += 1

        if limit_per_class > 0 and all(
            counters[cls] >= limit_per_class for cls in ALL_CLASSES
        ):
            print(f"[{split}] reached limit {limit_per_class} for all classes; stop reading remaining files.")
            break

    print(f"[{split}] total saved images: {total_written}")
    for cls in CLASS_ORDER:
        print(f"  {cls}: {counters[cls]}")


def main():
    args = parse_args()
    if args.dst.exists():
        print(f"Destination {args.dst} already exists. It will be overwritten.")
    args.dst.mkdir(parents=True, exist_ok=True)

    convert_split(
        split="train",
        files=SPLIT_FILES["train"],
        src_dir=args.src,
        dst_dir=args.dst,
        batch_size=args.batch_size,
        limit_per_class=args.train_limit,
    )
    convert_split(
        split="test",
        files=SPLIT_FILES["test"],
        src_dir=args.src,
        dst_dir=args.dst,
        batch_size=args.batch_size,
        limit_per_class=args.test_limit,
    )
    print("Conversion finished.")


if __name__ == "__main__":
    main()
