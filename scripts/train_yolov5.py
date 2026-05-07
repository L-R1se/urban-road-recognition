"""
YOLOv5 Training Script for Traffic Sign Detection.

Trains a YOLOv5-nano model on the custom 7-class dataset.

Usage:
    python train_yolov5.py --data dataset.yaml --weights yolov5n.pt --epochs 300 --img 320 --batch 32
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML config')
    parser.add_argument('--weights', type=str, default='yolov5n.pt',
                        help='Initial weights (yolov5n.pt for nano)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--img', type=int, default=320,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (0 for GPU)')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='urban_road')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    return parser.parse_args()


def main():
    args = parse_args()

    # Training uses the Ultralytics YOLOv5 training pipeline.
    # Clone https://github.com/ultralytics/yolov5 and run:
    #
    #   python train.py --data <dataset.yaml> --weights yolov5n.pt \
    #       --epochs 300 --img 320 --batch 32 --device 0
    #
    # The custom 7-class dataset should be structured as:
    #   dataset/
    #   ├── images/
    #   │   ├── train/
    #   │   └── val/
    #   ├── labels/
    #   │   ├── train/
    #   │   └── val/
    #   └── dataset.yaml
    #
    # Classes (0-indexed):
    #   0: danger (cone)
    #   1: side_walk (crosswalk)
    #   2: speed (speed limit lifted)
    #   3: speed_limit
    #   4: turn_left
    #   5: turn_right
    #   6: lane_change

    print(f"Training configuration:")
    print(f"  Data: {args.data}")
    print(f"  Weights: {args.weights}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.img}")
    print(f"  Batch size: {args.batch}")
    print(f"  Device: {args.device}")


if __name__ == "__main__":
    main()
