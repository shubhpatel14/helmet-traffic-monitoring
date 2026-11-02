# Indian Vehicle Number Plate Detection and Recognition Module

## 1. Module Overview
This module is a dedicated computer vision sub-system for the **Autonomous Helmet Detection and Traffic Compliance Monitoring System for Smart City**. It provides end-to-end detection and recognition of Indian vehicle registration plates using a fine-tuned **Ultralytics YOLOv8n** detection model and **EasyOCR** optical character recognition.

The module is architected to operate both as an autonomous command-line utility and as an integrated micro-service within the FastAPI backend, communicating with PostgreSQL storage and the monitoring dashboard.

---

## 2. Architecture & Workflow

```
       Traffic Image / CCTV Frame
                   │
                   ▼
     ┌────────────────────────────┐
     │  YOLOv8n Number Plate Det  │
     │   (license_plate_yolov8n)  │
     └─────────────┬──────────────┘
                   │ Bounding Boxes & Confidence
                   ▼
     ┌────────────────────────────┐
     │ Coordinate Bounds Clipping │
     │  & Min-Dimension Filtering │
     └─────────────┬──────────────┘
                   │ Valid License Plate Crop(s)
                   ▼
     ┌────────────────────────────┐
     │ Crop Preprocessing (CLAHE) │
     │  & Aspect Ratio Upscaling  │
     └─────────────┬──────────────┘
                   │ High-Contrast Crop
                   ▼
     ┌────────────────────────────┐
     │     EasyOCR Text Reader    │
     │      (Latin / English)     │
     └─────────────┬──────────────┘
                   │ Raw Token Transcriptions
                   ▼
     ┌────────────────────────────┐
     │ Indian Plate Normalization │
     │ (State Codes, BH, Strict)  │
     └─────────────┬──────────────┘
                   │
                   ▼
     ┌────────────────────────────┐
     │ Final Structured Result    │
     │ (JSON / Dict / Annotated)  │
     └────────────────────────────┘
```

---

## 3. Technology Stack
- **Deep Learning Framework**: PyTorch (`torch >= 2.1.0`), Torchvision (`torchvision >= 0.16.0`)
- **Object Detection**: Ultralytics YOLOv8 (`ultralytics >= 8.0.0`)
- **Optical Character Recognition**: EasyOCR (`easyocr >= 1.7.0`)
- **Image Processing**: OpenCV (`opencv-python >= 4.8.0`), NumPy
- **Backend Framework**: FastAPI, Pydantic, SQLAlchemy, Uvicorn
- **Testing**: Pytest

---

## 4. Installation & Setup

1. **Activate Virtual Environment** (Windows):
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Install Module Dependencies**:
   ```powershell
   pip install -r requirements.txt
   pip install -r backend/requirements.txt pytest
   ```

---

## 5. Model Weights Requirements
- **Weight Location**: `number_plate/weights/license_plate_yolov8n.pt`
- **File Size**: `6,248,291` bytes (~6.25 MB)
- **Automatic Fallback**: If the weight file is missing from `number_plate/weights/`, the module automatically retrieves the verified checkpoint from HuggingFace on first execution.
- **Hardware Acceleration**: The detector and OCR engines dynamically select CUDA GPU (`device="cuda:0"`) when hardware is available, otherwise gracefully falling back to CPU inference.

---

## 6. How to Run

### 6.1 Run YOLOv8 Plate Detection
```powershell
python number_plate/detector.py number_plate/test_images/car_04.jpg --conf 0.25
```
Example Output:
```text
[1] Plate Box: [1557, 1066, 1763, 1161] | Confidence: 0.695 | Crop Size: 206x95 px
```

### 6.2 Run OCR on a Cropped Plate Image
```powershell
python number_plate/ocr.py number_plate/results/cropped/car_04_plate_1.jpg
```
Example Output:
```text
Status:          recognized
Raw Text:        'GJLIC " 5555'
Normalized Text: GJLIC5555
OCR Confidence:  0.574
```

### 6.3 Run End-to-End Pipeline on a Single Traffic Frame
```powershell
# Standard Console Output
python number_plate/pipeline.py number_plate/test_images/bike_02.jpg --conf 0.15

# Formatted JSON Output
python number_plate/pipeline.py number_plate/test_images/bike_02.jpg --conf 0.15 --json

# Annotated Image & Saved Crops
python number_plate/pipeline.py number_plate/test_images/bike_02.jpg --conf 0.15 --output output.jpg --save-crops ./saved_crops
```

### 6.4 Run Full Genuine Image Benchmark Suite
```powershell
python number_plate/run_evaluation.py
```
This processes all 12 genuine test images, generates annotated bounding-box visual evidence in `number_plate/results/annotated/`, saves cropped plates in `number_plate/results/cropped/`, and exports metrics to `test_report.csv`, `test_report.json`, and `test_summary.md`.

### 6.5 Run Automated Test Suite
```powershell
pytest number_plate/tests/test_number_plate.py -v
```

---

## 7. Backend Integration & API Usage

### 7.1 Dedicated Number Plate Endpoint
- **URL**: `POST /api/number-plates/recognize`
- **Request Parameters**:
  - `file`: Multipart image file
  - `conf_threshold` (optional, default `0.15`)
  - `ocr_conf_threshold` (optional, default `0.20`)
- **Curl Example**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/number-plates/recognize" \
       -F "file=@number_plate/test_images/bike_02.jpg" \
       -F "conf_threshold=0.15"
  ```
- **Response Structure**:
  ```json
  {
    "total_detected": 1,
    "number_plate_results": [
      {
        "plate_text": "GJINC2394",
        "raw_ocr_text": "GJINC 2394_",
        "detector_confidence": 0.825,
        "ocr_confidence": 0.566,
        "plate_box": [571, 612, 1142, 768],
        "status": "recognized"
      }
    ]
  }
  ```

### 7.2 Integrated Helmet & Plate Analysis Endpoint
- **URL**: `POST /api/detections/analyze`
- Automatically invokes number plate detection alongside helmet compliance monitoring, persists the record into PostgreSQL/SQLite, annotates evidence, and provides live telemetry to the web dashboard.

---

## 8. Output Data Contract
Every detected license plate produces a structured Python dictionary / JSON object:
```json
{
  "plate_text": "string or null",
  "raw_ocr_text": "string",
  "detector_confidence": 0.825,
  "ocr_confidence": 0.566,
  "plate_box": [571, 612, 1142, 768],
  "status": "recognized or unreadable"
}
```

---

## 9. Known Limitations
1. **Low-Resolution Far-Distance Plates**: License plates located more than 20 meters from standard 720p/1080p camera frames occupy fewer than 20 vertical pixels, causing OCR character degradation.
2. **Ambiguous Character Substitution**: To avoid falsifying evidence, the normalization engine strictly rejects guessing characters. In standard Indian fonts, visually indistinguishable pairs like `1` vs `I`, `0` vs `O`, and `B` vs `8` may produce non-exact string matches unless verified against regional VAHAN database records.
3. **Dirty / Obstructed Plates**: Physical dirt, security rivets through digits, or custom designer scripts common on Indian vehicles are categorized as `unreadable` with `plate_text: null`.
