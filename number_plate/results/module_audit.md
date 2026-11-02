# Number Plate Module Audit Report

## 1. Executive Summary
This audit inspects the existing license plate detection and optical character recognition (OCR) sub-system located in `number_plate/`. The current codebase provides a solid initial foundation integrating **Ultralytics YOLOv8n** for plate localization and **EasyOCR** for text transcription. However, key structural gaps, missing configuration parameters, hardcoded CPU execution, lack of image validation/preprocessing, and unstructured output formats must be addressed before genuine testing and backend integration.

---

## 2. Model Weight Verification
- **Model Path**: `number_plate/weights/license_plate_yolov8n.pt`
- **File Existence**: Verified (`True`)
- **File Size**: `6,248,291` bytes (~6.25 MB)
- **Ultralytics Checkpoint Integrity**:
  - Model architecture: `ultralytics.nn.tasks.DetectionModel`
  - Model task: `detect`
  - Detection classes: `{0: 'license_plate'}`
  - Model load test: Successfully loaded without checkpoint corruption.
- **Inference Capability**:
  - CPU inference: Verified and fully functional.
  - GPU inference: Currently hardcoded to `device="cpu"` in detector; needs dynamic GPU fallback when CUDA is available (`device="cuda:0"` or `0` if available, else `"cpu"`).

---

## 3. Existing Functionality Audit

### 3.1 `number_plate/detector.py`
- **Current Role**: Implements `detect_plates(image_input, conf_threshold=0.25, weights_path=None)`.
- **Strengths**:
  - Validates input types (`str` vs `np.ndarray`).
  - Supports automated download fallback via `ensure_weights()`.
  - Performs lazy loading of the YOLO model.
  - Crops detected license plate bounding boxes with basic coordinate clipping to image bounds.
- **Identified Issues & Gaps**:
  1. **Hardcoded Device**: `model.predict(..., device="cpu")` prevents hardware acceleration on GPU-enabled environments.
  2. **Fixed Image Size**: Does not expose an `imgsz` parameter (defaults to 640), which limits optimization for higher or lower resolution camera streams.
  3. **Model Caching Inflexibility**: The global `_detector` caches the first loaded model; subsequent calls with a different `weights_path` ignore the parameter.
  4. **Crop Dimension Validation**: Degenerate boxes (e.g. 1x1 px or inverted boxes) are not validated against minimum practical dimensions (e.g., width >= 10, height >= 10).
  5. **CLI Output**: CLI prints basic box and confidence, but does not provide structured JSON or save individual cropped plates.

### 3.2 `number_plate/ocr.py`
- **Current Role**: Implements `read_plate_text(image_input)`.
- **Strengths**:
  - Lazy loads `easyocr.Reader(["en"])`.
  - Handles numpy arrays and file paths.
  - Strips leading/trailing whitespace from detected tokens.
- **Identified Issues & Gaps**:
  1. **Hardcoded CPU**: `easyocr.Reader(["en"], gpu=False)` ignores available CUDA devices.
  2. **No Image Preprocessing**: License plate crops from traffic cameras can have low contrast, blur, uneven lighting, or small resolution. The current code passes the raw crop directly to EasyOCR without contrast adjustment, grayscale conversion, or bilateral/adaptive filtering.
  3. **No Indian Registration Format Normalization**: Returns unstructured token list (`[{"text": ..., "confidence": ...}]`). It does not consolidate or normalize strings against standard Indian registration patterns (e.g., State Code + District Code + Series + Number: `DL8CAF5032`, `MH12DE1433`).
  4. **No Clean Separation of Raw vs. Normalized Text**: Lacks explicit fields distinguishing raw OCR output from validated/normalized text.
  5. **No Status Flag**: Does not return an explicit recognition status (`recognized` vs `unreadable`).

### 3.3 `number_plate/pipeline.py`
- **Current Role**: Chains `detect_plates()` and `read_plate_text()`.
- **Strengths**:
  - Simple invocation accepting path or image array.
  - Supports CLI execution with optional `--output` annotated image rendering.
- **Identified Issues & Gaps**:
  1. **Incompatible Output Schema**: Returns `{ "plate_box", "detector_confidence", "plate_text", "ocr_details" }`. The required project schema expects:
     ```json
     {
       "plate_text": "recognized text or null",
       "raw_ocr_text": "raw OCR output",
       "detector_confidence": 0.0,
       "ocr_confidence": 0.0,
       "plate_box": [x1, y1, x2, y2],
       "status": "recognized or unreadable"
     }
     ```
  2. **Confidence Handling**: Does not calculate an aggregated OCR confidence for the consolidated text string.
  3. **Unreadable Handling**: When OCR fails, `plate_text` is set to `""` instead of `None` / `null`, and no explicit `"status": "unreadable"` is assigned.
  4. **Multiple Plate Handling**: While it iterates, it does not distinguish multiple vehicle scenarios with structured per-plate metadata.
  5. **Robustness**: If an individual plate fails OCR or has an empty crop, the pipeline should gracefully report `"status": "unreadable"` without crashing.

---

## 4. Dependencies Verification
- `opencv-python`: `5.0.0.93` (Verified)
- `ultralytics`: `8.4.198` (Verified)
- `easyocr`: `1.7.2` (Verified)
- `torch`: `2.15.0+cpu` (Verified)
- `torchvision`: `0.29.2+cpu` (Verified)
- `pytest`: `9.2.0` (Verified)

---

## 5. Required Modifications Plan
1. **`number_plate/detector.py`**:
   - Add dynamic device detection (`device: Optional[str] = None` defaulting to CUDA if available, else CPU).
   - Add configurable `imgsz` parameter (default `640`).
   - Validate crop dimensions (minimum height and width threshold to prevent degenerate crops).
   - Ensure clean bounding box clipping and integer coordinate casting.
2. **`number_plate/ocr.py`**:
   - Support dynamic GPU/CPU selection for `easyocr.Reader`.
   - Add adaptive image preprocessing (contrast enhancement / CLAHE / grayscale conversion / upscale for small crops).
   - Implement conservative Indian registration normalization:
     - Remove special characters and spaces (`DL-08-CA-5032` -> `DL08CA5032`).
     - Validate against Indian number plate patterns (e.g. `^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$`, BH series `^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$`).
     - Never fabricate characters or make blind ambiguous replacements.
   - Return structured dict: `raw_text`, `normalized_text`, `ocr_confidence`, and `status`.
3. **`number_plate/pipeline.py`**:
   - Align return dictionary exactly with project specification.
   - Gracefully handle empty detections, unreadable crops, and multiple plates.
   - Provide helper for annotated image generation with bounding boxes, labels, and cropped plates directory output.
