# 🚗 Urban Road Recognition — Autonomous Driving ROS Smart Car

[![Competition](https://img.shields.io/badge/Competition-27th%20China%20Robot%20%26%20AI%20Competition-blue)](https://github.com)
[![Award](https://img.shields.io/badge/Award-National%202nd%20Prize-gold)](https://github.com)
[![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen)](https://www.ros.org/)
[![YOLOv5](https://img.shields.io/badge/Model-YOLOv5--nano-orange)](https://github.com/ultralytics/yolov5)
[![Jetson](https://img.shields.io/badge/Platform-Jetson%20Nano-green)](https://developer.nvidia.com/embedded/jetson-nano)

A vision-based autonomous driving system that integrates traditional computer vision with deep learning (YOLOv5) on an embedded AI platform. **Won National 2nd Prize** in the 27th China Robot and Artificial Intelligence Competition — Urban Road Recognition Category.

## 🎯 Core Features

| Task | Method | Accuracy |
|------|--------|----------|
| Lane Line Detection | HSV + Morphological Ops + Hough Transform | — |
| Traffic Sign Recognition (7 classes) | YOLOv5-nano + TensorRT | 98%+ |
| Pedestrian Detection & Emergency Stop | YOLOv5-nano + TensorRT | 98.8% |
| Traffic Light Color Recognition | HSV Color Space + Contour Analysis | — |
| Cone Detection & Obstacle Avoidance | HSV + YOLOv5-nano | 97.9% |

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────┐
│                 USB Camera (640×480)              │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│           NVIDIA Jetson Nano (ROS)               │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Lane     │  │ YOLOv5    │  │ Traffic      │  │
│  │ Detection│  │ Inference │  │ Light/Color  │  │
│  │ (OpenCV) │  │ (TensorRT)│  │ Detection    │  │
│  └────┬─────┘  └─────┬─────┘  └──────┬───────┘  │
│       └──────────────┼───────────────┘           │
│                      ▼                           │
│            ┌─────────────────┐                   │
│            │  Finite State   │                   │
│            │  Machine (FSM)  │                   │
│            └────────┬────────┘                   │
└─────────────────────┼───────────────────────────┘
                      ▼ (Serial / UART)
┌─────────────────────────────────────────────────┐
│          STM32F407VET6 (RTOS)                    │
│    ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│    │ Motor    │  │ Servo    │  │ Encoder/IMU │  │
│    │ Driver   │  │ Control  │  │ Feedback    │  │
│    └──────────┘  └──────────┘  └─────────────┘  │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│        4-Wheel Differential Chassis              │
└─────────────────────────────────────────────────┘
```

## 🔧 Hardware

| Component | Specification |
|-----------|---------------|
| Compute | NVIDIA Jetson Nano (4GB) |
| Microcontroller | STM32F407VET6 (ARM Cortex-M4) |
| Camera | USB Monocular Camera (640×480 @30fps) |
| Motor Driver | AT8236 |
| Servo | MG996 |
| Chassis | 4-Wheel Differential Drive |
| Power | 12V Li-Po Battery + Step-Down (5V) |

## 🚀 Key Optimizations

- **TensorRT Acceleration**: YOLOv5-nano inference on Jetson Nano achieves ~0.013s per frame (320×320), enabling real-time detection at 75+ FPS.
- **Lightweight Model**: YOLOv5-nano with only 1.9M parameters, balancing accuracy and speed on edge hardware.
- **Anti-Interference Pipeline**: Gaussian blur → Erosion → Dilation → Mask filtering for robust lane detection under varying lighting.
- **Data Augmentation**: Albumentations-based pixel-level and spatial-level augmentation (brightness, contrast, blur, affine transform) to improve model generalization.
- **Multi-ROI Strategy**: Different Regions of Interest for different sign positions to reduce false positives and speed up inference.

## 📂 Repository Structure

```
├── README.md                   # Project overview
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── src/
│   ├── main.py                 # Main ROS node (entry point)
│   ├── lane_detection.py       # Lane line detection pipeline
│   ├── object_detection.py     # YOLOv5 + TensorRT inference
│   ├── traffic_light.py        # Traffic light color recognition
│   ├── cone_detection.py       # Cone detection & avoidance
│   ├── servo_control.py        # Servo PD controller
│   └── speed_control.py        # Speed FSM & control logic
├── scripts/
│   ├── train_yolov5.py         # YOLOv5 training script
│   └── convert_tensorrt.py     # ONNX → TensorRT engine conversion
├── stm32_firmware/             # STM32 RTOS-based firmware
├── models/                     # Trained model files (.engine)
├── dataset/                    # Dataset documentation & samples
└── docs/
    └── images/                 # System diagrams & test results
```

## 🏁 Quick Start

### Prerequisites

- NVIDIA Jetson Nano with JetPack 4.6+
- ROS Noetic
- Python 3.8+
- TensorRT 8.0+
- OpenCV 4.5+ with CUDA support

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/urban-road-recognition.git
cd urban-road-recognition

# Install dependencies
pip install -r requirements.txt

# Build ROS workspace
catkin_make
source devel/setup.bash

# Run the system
rosrun opencv_detection main.py
```

### Training YOLOv5

```bash
cd scripts
python train_yolov5.py --data ../dataset/dataset.yaml --weights yolov5n.pt --epochs 300 --img 320
python convert_tensorrt.py --weights best.pt --output ../models/best.engine
```

## 📊 Test Results

| Metric | Value |
|--------|-------|
| Pedestrian Detection Accuracy | 98.8% |
| Pedestrian Stop Reaction Time | ~0.5s |
| Cone Detection Accuracy | 97.9% |
| Model Inference Time (TensorRT) | ~0.013s/frame |
| System Stability | 3+ hours continuous operation |
| FPS (end-to-end) | 30+ fps |

## 🏆 Competition Results

- **Competition**: 27th China Robot and Artificial Intelligence Competition
- **Category**: Urban Road Recognition
- **Award**: National 2nd Prize

## 🔬 Technical Report

See the full technical report (in Chinese) for detailed methodology, formulas, and analysis.

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection framework
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for inference optimization
- [OpenCV](https://opencv.org/) for computer vision utilities
- [ROS](https://www.ros.org/) for the robotics middleware
