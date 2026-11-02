import argparse
import json
import os
import sys
from typing import Optional, Union

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from number_plate.detector import detect_plates
    from number_plate.ocr import recognize_plate_text
else:
    from .detector import detect_plates
    from .ocr import recognize_plate_text


def process_traffic_image(
    image_input: Union[str, np.ndarray],
    conf_threshold: float = 0.25,
    ocr_conf_threshold: float = 0.20,
    weights_path: Optional[str] = None,
    imgsz: int = 640,
    device: Optional[str] = None,
    include_crop: bool = False,
) -> list[dict]:
    """Execute the end-to-end number plate pipeline: YOLO detection followed by EasyOCR.

    Args:
        image_input: Path to traffic image (str) or loaded BGR image (np.ndarray).
        conf_threshold: Detector confidence threshold.
        ocr_conf_threshold: Minimum confidence threshold for OCR token acceptance.
        weights_path: Optional custom YOLO weights path.
        imgsz: Inference image size.
        device: Device to run inference on ('cpu', 'cuda:0', or None for auto).
        include_crop: Whether to include the cropped numpy ndarray in output dict.

    Returns:
        List of dicts, each strictly formatted as:
            {
                "plate_text": str or None,
                "raw_ocr_text": str,
                "detector_confidence": float,
                "ocr_confidence": float,
                "plate_box": [x1, y1, x2, y2],
                "status": "recognized" or "unreadable"
            }
    """
    try:
        detected_plates = detect_plates(
            image_input,
            conf_threshold=conf_threshold,
            weights_path=weights_path,
            imgsz=imgsz,
            device=device,
        )
    except Exception as exc:
        print(f"Warning: Plate detection encountered an error: {exc}", file=sys.stderr)
        return []

    results: list[dict] = []
    for plate in detected_plates:
        crop = plate.get("crop")
        box = plate.get("box", [])
        det_conf = float(plate.get("confidence", 0.0))

        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            item = {
                "plate_text": None,
                "raw_ocr_text": "",
                "detector_confidence": det_conf,
                "ocr_confidence": 0.0,
                "plate_box": box,
                "status": "unreadable",
            }
            if include_crop:
                item["crop"] = crop
            results.append(item)
            continue

        try:
            ocr_res = recognize_plate_text(
                crop, conf_threshold=ocr_conf_threshold
            )
            plate_text = ocr_res.get("normalized_text")
            raw_text = ocr_res.get("raw_text", "")
            ocr_conf = float(ocr_res.get("ocr_confidence", 0.0))
            status = ocr_res.get("status", "unreadable")
        except Exception as ocr_err:
            print(f"Warning: OCR failed on detected plate crop: {ocr_err}", file=sys.stderr)
            plate_text = None
            raw_text = ""
            ocr_conf = 0.0
            status = "unreadable"

        item = {
            "plate_text": plate_text,
            "raw_ocr_text": raw_text,
            "detector_confidence": det_conf,
            "ocr_confidence": ocr_conf,
            "plate_box": box,
            "status": status,
        }
        if include_crop:
            item["crop"] = crop

        results.append(item)

    return results


def draw_pipeline_annotations(
    image: np.ndarray,
    results: list[dict],
) -> np.ndarray:
    """Draw bounding boxes and formatted text labels onto image for visualization."""
    annotated = image.copy()
    for item in results:
        box = item["plate_box"]
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        status = item.get("status", "unreadable")
        det_conf = item.get("detector_confidence", 0.0)
        ocr_conf = item.get("ocr_confidence", 0.0)
        plate_text = item.get("plate_text")

        if status == "recognized" and plate_text:
            color = (0, 200, 0)  # Green for recognized
            label = f"{plate_text} (D:{det_conf:.2f} O:{ocr_conf:.2f})"
        else:
            color = (0, 165, 255)  # Orange for unreadable
            label = f"Unreadable ({det_conf:.2f})"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y1 = max(y1 - txt_h - 10, 0)
        label_y2 = label_y1 + txt_h + 8
        label_x2 = min(x1 + txt_w + 6, annotated.shape[1])

        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return annotated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end traffic number-plate detector and OCR reader."
    )
    parser.add_argument("image_path", help="Path to traffic or vehicle image")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.20,
        help="OCR minimum confidence threshold (default: 0.20)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Detection image size (default: 640)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results formatted as JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save annotated output image with detected boxes and text",
    )
    parser.add_argument(
        "--save-crops",
        type=str,
        default=None,
        help="Optional directory to save cropped plate images",
    )
    args = parser.parse_args()

    try:
        pipeline_results = process_traffic_image(
            args.image_path,
            conf_threshold=args.conf,
            ocr_conf_threshold=args.ocr_conf,
            imgsz=args.imgsz,
            include_crop=bool(args.save_crops),
        )

        display_results = [
            {k: v for k, v in item.items() if k != "crop"} for item in pipeline_results
        ]

        if args.json:
            print(json.dumps(display_results, indent=2))
        else:
            if not display_results:
                print("No number plates detected.")
            else:
                for idx, item in enumerate(display_results, start=1):
                    plate_display = item["plate_text"] or "(unreadable)"
                    print(
                        f"[{idx}] Status: {item['status']:10s} | "
                        f"Box: {item['plate_box']} | "
                        f"Det Conf: {item['detector_confidence']:.3f} | "
                        f"OCR Conf: {item['ocr_confidence']:.3f} | "
                        f"Text: '{plate_display}' | "
                        f"Raw: '{item['raw_ocr_text']}'"
                    )

        if args.save_crops and pipeline_results:
            os.makedirs(args.save_crops, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            for idx, item in enumerate(pipeline_results, start=1):
                crop_img = item.get("crop")
                if crop_img is not None and crop_img.size > 0:
                    crop_filename = f"{base_name}_plate_{idx}.jpg"
                    crop_dest = os.path.join(args.save_crops, crop_filename)
                    cv2.imwrite(crop_dest, crop_img)
                    print(f"Saved crop: {crop_dest}")

        if args.output and pipeline_results:
            orig_img = cv2.imread(args.image_path)
            if orig_img is not None:
                annotated_img = draw_pipeline_annotations(orig_img, display_results)
                out_dir = os.path.dirname(os.path.abspath(args.output))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(args.output, annotated_img)
                print(f"Annotated image saved to: {args.output}")
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
