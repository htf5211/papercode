# PaperCode: Rail Defect Dataset Quality Assessment

This repository contains the dataset-quality evaluation code used for rail-surface defect detection research. The framework evaluates a YOLO-format dataset from four complementary dimensions:

- **Spatial characteristics** (`spatial.py`)
- **Optical characteristics** (`light.py`)
- **Style consistency** (`style.py`)
- **Annotation consistency** (`annotations.py`)

The four component scores are aggregated with a geometric mean in `code/main.py`.

## Installation

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Dataset layout

The code expects a YOLO-format split:

```text
dataset/
├── images/
│   ├── image_001.jpg
│   └── ...
└── labels/
    ├── image_001.txt
    └── ...
```

Each label line should follow the standard format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`.

## Usage

```bash
python code/main.py --images /path/to/images --labels /path/to/labels
```

On Windows:

```powershell
python code/main.py --images "D:\dataset\train\images" --labels "D:\dataset\train\labels"
```

The annotation metrics use a pretrained ResNet feature extractor. The first run may download the official torchvision weights if they are not already cached.

## Files

| File | Purpose |
|---|---|
| `code/main.py` | Command-line entry point and score aggregation |
| `code/spatial.py` | Spatial structure and information metrics |
| `code/light.py` | Exposure, illumination, blur, and high-frequency metrics |
| `code/style.py` | Color, tone, texture, and noise/compression consistency metrics |
| `code/annotations.py` | Annotation consistency metrics |
| `code/utils.py` | Shared image and label utilities |
| `DATASETS.md` | Source links and redistribution notes for the ten datasets |

## Notes

- Model checkpoints, training runs, and caches are intentionally excluded.
- The ten datasets are not mirrored in this repository. See [`DATASETS.md`](DATASETS.md) for their original source pages and licenses.
- Metric definitions and parameter settings should be cited and reported consistently with the accompanying paper.

## Citation

Citation information will be added after the paper is published.

## License

No license has been selected yet. All rights are reserved unless a license file is added.
