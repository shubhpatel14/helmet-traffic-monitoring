import argparse
import os
import sys
import urllib.request
from typing import Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "license_plate_yolov8n.pt")
MODEL_URL = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"

_cached_detectors: dict[str, YOLO] = {}


def _select_device(device: Optional[str] = None) -> str:
    """Select inference device (CUDA if available and requested, otherwise CPU)."""
    if device is not None:
        return str(device)
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_weights(weights_path: str = DEFAULT_WEIGHTS_PATH) -> str:
    """Ensure that the pretrained license plate detector weights are present locally."""
    is_default = os.path.abspath(weights_path) == os.path.abspath(DEFAULT_WEIGHTS_PATH)
    if is_default:
        if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1000000:
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            print(f"Downloading detector weights to {weights_path}...")
            urllib.request.urlretrieve(MODEL_URL, weights_path)
    else:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Specified weights file not found: {weights_path}")
    return weights_path


def _get_detector(weights_path: Optional[str] = None) -> YOLO:
    """Lazily load and cache the YOLO license plate detector by weights path."""
    target_path = ensure_weights(weights_path or DEFAULT_WEIGHTS_PATH)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"YOLO weights file not found: {target_path}")

    norm_path = os.path.abspath(target_path)
    if norm_path not in _cached_detectors:
        _cached_detectors[norm_path] = YOLO(norm_path)
    return _cached_detectors[norm_path]


def detect_plates(
    image_input: Union[str, np.ndarray],
    conf_threshold: float = 0.25,
    weights_path: Optional[str] = None,
    imgsz: int = 640,
    device: Optional[str] = None,
    min_size: tuple[int, int] = (10, 10),
) -> list[dict]:
    """Detect license plates within an image or full traffic frame.

    Args:
        image_input: Path to the image file (str) or loaded BGR image (np.ndarray).
        conf_threshold: Detection confidence threshold (0.0 to 1.0).
        weights_path: Optional path to custom YOLO weights.
        imgsz: Inference image size for YOLOv8 (default: 640).
        device: Device to run inference on (e.g. 'cpu', 'cuda:0', or None for auto).
        min_size: Minimum crop width and height in pixels (width, height).

    Returns:
        List of dicts with:
            - 'box': [x1, y1, x2, y2] bounding box coordinates (integers)
            - 'confidence': detection confidence score rounded to 3 decimal places
            - 'crop': cropped license plate region as a numpy ndarray (BGR)

    Raises:
        FileNotFoundError: If the image path does not exist or cannot be read.
        TypeError: If image_input is not a path string or numpy ndarray.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file does not exist: {image_input}")
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Unable to read or decode image at: {image_input}")
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            return []
        image = image_input
    else:
        raise TypeError(
            f"Expected image path string or np.ndarray, got {type(image_input).__name__}"
        )

    model = _get_detector(weights_path)
    chosen_device = _select_device(device)
    img_h, img_w = image.shape[:2]
    if img_h == 0 or img_w == 0:
        return []

    # Run inference with configurable confidence, imgsz, and device
    results = model.predict(
        source=image,
        conf=conf_threshold,
        imgsz=imgsz,
        device=chosen_device,
        verbose=False,
    )

    detections: list[dict] = []
    if not results or len(results) == 0:
        return detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    min_w, min_h = min_size
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        x1 = max(0, min(int(round(xyxy[0])), img_w - 1))
        y1 = max(0, min(int(round(xyxy[1])), img_h - 1))
        x2 = max(x1 + 1, min(int(round(xyxy[2])), img_w))
        y2 = max(y1 + 1, min(int(round(xyxy[3])), img_h))

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < min_w or crop_h < min_h:
            continue

        crop = image[y1:y2, x1:x2].copy()
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            continue

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "crop": crop,
            }
        )

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect license plates in an image using pretrained YOLOv8n."
    )
    parser.add_argument("image_path", help="Path to input traffic or vehicle image")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for plate detection (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g. cpu, cuda:0, or None for auto)",
    )
    args = parser.parse_args()

    try:
        plates = detect_plates(
            args.image_path,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
        if not plates:
            print("No license plates detected.")
        else:
            for idx, p in enumerate(plates, start=1):
                crop_shape = p["crop"].shape
                print(
                    f"[{idx}] Plate Box: {p['box']} | Confidence: {p['confidence']} | Crop Size: {crop_shape[1]}x{crop_shape[0]} px"
                )
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
