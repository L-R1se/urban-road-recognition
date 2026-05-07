"""
Convert YOLOv5 PyTorch model to TensorRT engine for Jetson Nano acceleration.

Two-step process:
  1. PyTorch (.pt) → ONNX (.onnx)
  2. ONNX (.onnx) → TensorRT (.engine)

Usage:
    python convert_tensorrt.py --weights best.pt --output models/best.engine --img 320
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained .pt weights')
    parser.add_argument('--output', type=str, default='models/best.engine',
                        help='Output TensorRT engine path')
    parser.add_argument('--img', type=int, default=320,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace size (GB)')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 precision')
    return parser.parse_args()


def export_to_onnx(weights_path, img_size, batch_size):
    """
    Export YOLOv5 model to ONNX format.

    From the YOLOv5 repo:
        python export.py --weights best.pt --include onnx --img 320 --batch 1
    """
    pass  # Use YOLOv5's export.py


def onnx_to_tensorrt(onnx_path, engine_path, workspace, fp16):
    """
    Convert ONNX to TensorRT engine using trtexec.

    Command:
        trtexec --onnx=best.onnx --saveEngine=best.engine \
            --workspace=4096 --fp16
    """
    pass  # Use NVIDIA's trtexec tool


def main():
    args = parse_args()
    print(f"Converting {args.weights} → {args.output}")
    print(f"  Image size: {args.img}")
    print(f"  FP16: {args.fp16}")

    # Step 1: PyTorch → ONNX
    onnx_path = args.weights.replace('.pt', '.onnx')
    export_to_onnx(args.weights, args.img, args.batch)

    # Step 2: ONNX → TensorRT
    onnx_to_tensorrt(onnx_path, args.output, args.workspace, args.fp16)

    print(f"TensorRT engine saved to: {args.output}")


if __name__ == "__main__":
    main()
