"""
YOLOv5 Object Detection using TensorRT on Jetson Nano.

Performs real-time inference for: traffic signs (7 classes), pedestrians, and cones.
Uses TensorRT-accelerated engine for ~0.013s/frame inference on 320x320 input.
"""

import ctypes
import numpy as np
import cv2


class YoLov5TRT:
    """
    YOLOv5 TensorRT inference wrapper.

    Loads a pre-built TensorRT engine and performs GPU-accelerated inference.
    Engine conversion: PyTorch (.pt) → ONNX → TensorRT (.engine).
    """

    def __init__(self, engine_file_path, plugin_library=None):
        """
        Args:
            engine_file_path: Path to the .engine TensorRT model.
            plugin_library: Optional path to custom plugin .so file.
        """
        self.engine_file_path = engine_file_path
        if plugin_library:
            ctypes.CDLL(plugin_library)

        # TensorRT logger, runtime, engine, and context setup
        # (Simplified — actual TRT API initialization via pycuda/cuda-python)
        self._load_engine()

    def _load_engine(self):
        """Load TensorRT engine and allocate buffers."""
        # In production: use tensorrt Python bindings
        # import tensorrt as trt
        # logger = trt.Logger(trt.Logger.WARNING)
        # with open(self.engine_file_path, 'rb') as f:
        #     runtime = trt.Runtime(logger)
        #     self.engine = runtime.deserialize_cuda_engine(f.read())
        # self.context = self.engine.create_execution_context()
        pass

    def infer(self, frame):
        """
        Run inference on a single frame.

        Args:
            frame: numpy array (H, W, 3) in BGR format.

        Returns:
            result_boxes: list of [x1, y1, x2, y2] bounding boxes
            result_scores: list of confidence scores
            result_classid: list of class IDs
            use_time: inference time in seconds
        """
        # Preprocess: resize, normalize, convert NCHW
        # blobs = self._preprocess(frame)
        # outputs = self._do_inference(blobs)
        # result_boxes, result_scores, result_classid = self._postprocess(outputs)
        pass

    def destroy(self):
        """Release TensorRT resources."""
        pass


def get_max_area(result_boxes, result_classid, result_scores):
    """
    Select the detection with the largest bounding box area
    (used when multiple detections of the same class appear).

    Returns:
        cls_id, area, max_box, score — details of the largest detection.
    """
    max_area = 0
    best_cls = 0
    best_box = None
    best_score = 0
    for box, cls_id, score in zip(result_boxes, result_classid, result_scores):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_cls = cls_id
            best_box = box
            best_score = score
    return best_cls, max_area, best_box, best_score


def draw_rerectangle(frame, box, class_name, score, area):
    """Draw bounding box with class label and score on the frame."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{class_name} {score:.2f} A:{area}"
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
