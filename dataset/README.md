# Dataset Documentation

## Overview

Custom dataset for urban road recognition, captured from the vehicle's perspective using the onboard USB camera. Contains 7 classes of traffic signs, pedestrians, and obstacles.

## Classes

| ID | Name | Description | Samples |
|----|------|-------------|---------|
| 0 | danger | Traffic cone / obstacle | ~500 |
| 1 | side_walk | Crosswalk / pedestrian crossing | ~500 |
| 2 | speed | Speed limit lifted | ~500 |
| 3 | speed_limit | Speed limit sign | ~500 |
| 4 | turn_left | Left turn sign | ~500 |
| 5 | turn_right | Right turn sign | ~500 |
| 6 | lane_change | Lane change sign | ~500 |

## Data Collection

- **Camera**: USB monocular camera, 640×480 resolution
- **Platform**: Moving vehicle (Jetson Nano + 4-wheel chassis)
- **Environments**: Indoor simulated urban roads with varying lighting conditions
- **Scenarios**: Multiple track materials (mesh fabric, spray-painted cloth)

## Data Augmentation

Using the [Albumentations](https://github.com/albumentations-team/albumentations) library:

- **Pixel-level**: Brightness adjustment, contrast change, blur, histogram equalization
- **Spatial-level**: Scaling, rotation, affine transformation, perspective shift

## Dataset Format

YOLO format:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       ├── img_701.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   └── ...
│   └── val/
│       ├── img_701.txt
│       └── ...
└── dataset.yaml
```

`dataset.yaml`:

```yaml
path: ../dataset
train: images/train
val: images/val

names:
  0: danger
  1: side_walk
  2: speed
  3: speed_limit
  4: turn_left
  5: turn_right
  6: lane_change
```

## Note

The actual images and labels are not included in this repository due to size constraints. Please contact the authors for access to the full dataset.
