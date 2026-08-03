from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from annotations import annotation_consistency_metrics
from light import light_metrics
from spatial import spatial_metrics
from style import style_consistency


def evaluate_dataset(images_dir: Path, labels_dir: Path) -> dict[str, float]:
    """Evaluate one YOLO-format dataset split and return all component scores."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    spatial_score = spatial_metrics(images_dir, labels_dir)
    light_score = light_metrics(images_dir, labels_dir)
    style_score = style_consistency(
        images_dir,
        resize_long_side=640,
        noise_sigma=1.2,
        noise_ksize=0,
        include_blockiness=True,
    )
    annotation_score = annotation_consistency_metrics(
        images_dir,
        labels_dir,
        pad_ratio=0.10,
        max_patches=None,
        batch_size=2,
        knn_k=10,
        knn_chunk_size=128,
        split_iou_thr=0.10,
        ring_ws=(2, 4, 8, 12),
        ring_q=0.30,
        ring_agg="median",
        b2_min_per_class=10,
    )

    components = np.asarray(
        [spatial_score, light_score, style_score, annotation_score],
        dtype=np.float64,
    )
    if np.any(components <= 0):
        raise ValueError("All component scores must be positive for geometric aggregation.")

    return {
        "spatial_score": float(spatial_score),
        "light_score": float(light_score),
        "style_score": float(style_score),
        "annotation_score": float(annotation_score),
        "dataset_score": float(np.exp(np.mean(np.log(components)))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the quality characteristics of a YOLO-format dataset split."
    )
    parser.add_argument("--images", type=Path, required=True, help="Image directory")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO label directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = evaluate_dataset(args.images.resolve(), args.labels.resolve())
    for name, value in scores.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
