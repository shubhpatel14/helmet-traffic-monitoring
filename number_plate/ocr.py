import argparse
import os
import re
import sys
from typing import Optional, Union

import cv2
import easyocr
import numpy as np
import torch

_reader = None

# Known Indian State and Union Territory 2-letter codes
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TR", "TS", "UK", "UA", "UP", "WB"
}

# Regex patterns for Indian vehicle registration formats
# Standard: DL08CAF5032, MH12DE1433, GJ01AB1234, HR26DQ5551
STANDARD_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
# Bharat (BH) Series: 22BH1234AA
BH_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
# Older / vintage / 3-digit number format: DL1C1234, DL8C123
VINTAGE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$")


def _get_reader(gpu: Optional[bool] = None) -> easyocr.Reader:
    """Lazily initialize and return the cached EasyOCR reader instance."""
    global _reader
    if gpu is None:
        gpu = torch.cuda.is_available()
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=gpu)
    return _reader


def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """Preprocess license plate crop to improve OCR accuracy."""
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    # Upscale small crops to ensure minimum character height
    target_h = max(h, 64)
    target_w = int(w * (target_h / max(h, 1)))
    if target_h != h or target_w != w:
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    else:
        resized = crop.copy()

    # Convert to grayscale if 3 channels
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def normalize_plate_text(raw_text: str) -> tuple[Optional[str], bool]:
    """Conservatively clean and validate recognized license plate text.

    Rules:
    - Remove non-alphanumeric noise characters and extra whitespace.
    - Do not blindly substitute ambiguous characters (e.g. O->0 or I->1).
    - Strip leading 'IND' country code identifier commonly found on HSRP plates.
    - Check against standard Indian registration patterns.

    Returns:
        tuple (normalized_text_or_none, is_valid_pattern)
    """
    if not raw_text:
        return None, False

    # Remove special punctuation and symbols, keep only alphanumeric
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

    # Remove leading 'IND' watermark if followed by a valid state code
    if cleaned.startswith("IND") and len(cleaned) > 5:
        sub_candidate = cleaned[3:]
        if sub_candidate[:2] in INDIAN_STATE_CODES or sub_candidate[:2].isdigit():
            cleaned = sub_candidate

    if len(cleaned) < 6:
        # Fewer than 6 characters is insufficient for a complete Indian registration
        return None, False

    # Validate against known Indian registration patterns
    is_standard = bool(STANDARD_PATTERN.match(cleaned))
    is_bh = bool(BH_PATTERN.match(cleaned))
    is_vintage = bool(VINTAGE_PATTERN.match(cleaned))
    is_valid_pattern = is_standard or is_bh or is_vintage

    # Check if the state code matches a known Indian jurisdiction
    state_code = cleaned[:2]
    state_valid = state_code in INDIAN_STATE_CODES or is_bh

    if is_valid_pattern and state_valid:
        return cleaned, True

    # If general length matches reasonable registration (6 to 11 chars) with state code
    if 6 <= len(cleaned) <= 11 and state_valid:
        return cleaned, False

    # If it has plausible format without matching strict state list
    if 6 <= len(cleaned) <= 10:
        return cleaned, False

    return None, False


def recognize_plate_text(
    image_input: Union[str, np.ndarray],
    conf_threshold: float = 0.20,
    gpu: Optional[bool] = None,
) -> dict:
    """Recognize number plate text with preprocessing, normalization, and confidence.

    Args:
        image_input: Path to cropped plate image or numpy BGR/grayscale array.
        conf_threshold: Minimum OCR token confidence to accept.
        gpu: Run on GPU if True, CPU if False, or auto-detect if None.

    Returns:
        dict containing:
            - 'raw_text': combined raw OCR transcription (str)
            - 'normalized_text': cleaned/validated plate number (str or None)
            - 'ocr_confidence': aggregate confidence score rounded to 3 decimal places
            - 'status': 'recognized' or 'unreadable'
            - 'tokens': list of detected raw tokens with local coordinates
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file does not exist: {image_input}")
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Unable to read or decode image at: {image_input}")
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            return {
                "raw_text": "",
                "normalized_text": None,
                "ocr_confidence": 0.0,
                "status": "unreadable",
                "tokens": [],
            }
        image = image_input
    else:
        raise TypeError(
            f"Expected image path string or np.ndarray, got {type(image_input).__name__}"
        )

    # Preprocess image crop for optimal character contrast
    preprocessed = preprocess_plate_crop(image)

    reader = _get_reader(gpu=gpu)
    # Allow alphanumeric characters
    raw_results = reader.readtext(preprocessed)

    if not raw_results:
        # Fallback to raw crop if preprocessed yielded nothing
        raw_results = reader.readtext(image)

    tokens: list[dict] = []
    text_parts: list[str] = []
    conf_scores: list[float] = []

    for bbox, text, conf in raw_results:
        clean_token = str(text).strip()
        conf_val = float(conf)
        if not clean_token or conf_val < conf_threshold:
            continue

        box_coords = [[int(round(coord)) for coord in point] for point in bbox]
        tokens.append(
            {
                "text": clean_token,
                "confidence": round(conf_val, 3),
                "box": box_coords,
            }
        )
        text_parts.append(clean_token)
        conf_scores.append(conf_val)

    if not tokens:
        return {
            "raw_text": "",
            "normalized_text": None,
            "ocr_confidence": 0.0,
            "status": "unreadable",
            "tokens": [],
        }

    raw_text = " ".join(text_parts).strip()
    avg_conf = round(float(np.mean(conf_scores)), 3)

    normalized_text, is_valid = normalize_plate_text(raw_text)

    # Determine status: recognized requires normalized text and reasonable confidence
    if normalized_text and avg_conf >= conf_threshold:
        status = "recognized"
    else:
        status = "unreadable"

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text if status == "recognized" else None,
        "ocr_confidence": avg_conf,
        "status": status,
        "tokens": tokens,
    }


def read_plate_text(image_input: Union[str, np.ndarray]) -> list[dict]:
    """Legacy compatibility function returning raw detected tokens."""
    recognition = recognize_plate_text(image_input)
    return recognition["tokens"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and normalize text from an image or cropped number plate."
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Minimum confidence threshold for OCR tokens (default: 0.20)",
    )
    args = parser.parse_args()

    try:
        result = recognize_plate_text(args.image_path, conf_threshold=args.conf)
        print(f"Status:          {result['status']}")
        print(f"Raw Text:        '{result['raw_text']}'")
        print(f"Normalized Text: {result['normalized_text']}")
        print(f"OCR Confidence:  {result['ocr_confidence']}")
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
