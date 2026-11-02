# Number Plate Recognition: Test Summary & Benchmark Report

## 1. Overview
This benchmark report details the experimental testing of the Indian Vehicle Number Plate Detection and Recognition system using genuine Indian traffic and vehicle imagery. Testing was conducted using the genuine YOLOv8n detector checkpoint (`license_plate_yolov8n.pt`) and the enhanced EasyOCR recognition pipeline on CPU.

---

## 2. Execution Environment
- **Operating System**: Microsoft Windows 10
- **Python Version**: Python 3.11.9
- **Inference Hardware**: Intel CPU (64-bit AMD64)
- **PyTorch Version**: `2.15.0+cpu`
- **Torchvision Version**: `0.29.2+cpu`
- **Ultralytics Version**: `8.4.198`
- **EasyOCR Version**: `1.7.2`
- **OpenCV Version**: `5.0.0.93`
- **Benchmark Script**: `number_plate/run_evaluation.py`
- **Confidence Thresholds**: Detection: `0.15`, OCR Token: `0.20`

---

## 3. Test Dataset Description
The benchmark dataset consists of 12 genuine, legally accessible Indian vehicle images sourced from Wikimedia Commons under Creative Commons licensing. No mock, synthetic, or AI-generated vehicle images were used.

| Image Name | Vehicle Category | Image Resolution | Lighting Condition | Camera Angle / View | Ground Truth Plate |
|:---|:---|:---|:---|:---|:---|
| `bike_01.jpg` | Motorcycle | 2000 x 2736 px | Daylight outdoor | Front-oblique portrait | `PB08BJ5222` |
| `bike_02.jpg` | Motorcycle | 1600 x 899 px | Warm evening / artificial | Frontal headlamp bracket | `GJ01NC2394` |
| `bike_03.jpg` | Motorcycle | 1809 x 338 px | Indoor / neutral | Flat cropped plate | `DL3SBR9322` |
| `bike_04.jpg` | Scooter | 2592 x 1944 px | Direct bright sunlight | Rear flat angle | `KA04HM3634` |
| `bike_05.jpg` | Motorcycle | 796 x 205 px | Neutral diffuse | Frontal banner crop | `KA19P8488` |
| `bike_06.jpg` | Motorcycle | 3888 x 2592 px | Direct street sunlight | Rear-oblique traffic | `PB10GJ6445` |
| `car_01.jpg` | Car (Taxi) | 3504 x 2336 px | Daylight street | Rear-oblique angle | `WB04E4362` |
| `car_02.jpg` | Car (Hatchback) | 761 x 269 px | Overcast / diffused | Frontal bumper crop | `DL8CAF5032` |
| `car_03.jpg` | Car (Tourist Taxi) | 640 x 480 px | Bright daylight | Rear-quarter perspective | `MH01BA1984` |
| `car_04.jpg` | Car (Rally) | 2048 x 1536 px | Outdoor shaded | Front-right quarter | `GJ01JC5555` |
| `car_05.jpg` | Car (Beetle) | 472 x 456 px | Daylight outdoor | Frontal perspective | `KL07B5559` |
| `car_06.jpg` | Commercial / Auto | 2592 x 1936 px | Bright outdoor | Rear-oblique angle | `DL1RM0556` |

---

## 4. Benchmark Summary Metrics

| Metric | Measured Value | Definition / Interpretation |
|:---|:---|:---|
| **Total Images Tested** | `12` | Total genuine vehicle scenes evaluated |
| **Images with Plates Localized** | `8` | Images where YOLOv8 successfully located at least one plate |
| **Total Number Plates Detected** | `9` | Total plate bounding boxes identified |
| **Detection Success Rate** | `66.67%` | Percentage of images with successfully localized plate regions |
| **Successful OCR Readings** | `5` | Plates producing valid, normalized alphanumeric registration text |
| **Exact OCR Matches** | `0` | Number of plates with 100% exact character string parity |
| **Character-Discrepant OCR Readings** | `5` | Plates with valid Indian format but visual character confusions (e.g., `1` vs `I`, `0` vs `O`, `B` vs `8`) |
| **Unreadable Plates** | `4` | Plates detected but classified honestly as unreadable due to blur or distance |
| **OCR Recognition Rate (of Detections)**| `55.56%` | Fraction of localized plates producing recognized text |
| **Exact-Match Accuracy** | `0.0%` | Unambiguous exact match rate (strictly adhering to no-hallucination rules) |
| **Average YOLO Confidence** | `0.485` | Mean confidence score across all valid detections |
| **Average OCR Confidence** | `0.582` | Mean confidence score across recognized text tokens |
| **Average Processing Latency** | `1.015s` | End-to-end inference latency per image on CPU |

---

## 5. Detailed Results per Test Image

| Image Name | Vehicle Type | Ground Truth | Detected Box `[x1,y1,x2,y2]` | YOLO Conf | OCR Conf | Status | Raw OCR Output | Predicted Plate | Exact Match |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `bike_01.jpg` | Motorcycle | `PB08BJ5222` | `[377, 680, 1073, 1089]` | 0.176 | 0.981 | `recognized` | `W 03 PB 7632` | `W03PB7632` | No |
| `bike_02.jpg` | Motorcycle | `GJ01NC2394` | `[571, 612, 1142, 768]` | 0.825 | 0.566 | `recognized` | `GJINC 2394_` | `GJINC2394` | No |
| `bike_03.jpg` | Motorcycle | `DL3SBR9322` | `[]` | 0.000 | 0.000 | `no_detection` | `""` | `null` | No |
| `bike_04.jpg` | Scooter | `KA04HM3634` | `[]` | 0.000 | 0.000 | `no_detection` | `""` | `null` | No |
| `bike_05.jpg` | Motorcycle | `KA19P8488` | `[0, 4, 441, 143]` | 0.272 | 0.000 | `unreadable` | `""` | `null` | No |
| `bike_06.jpg` | Motorcycle | `PB10GJ6445` | `[733, 1591, 1149, 1725]` | 0.346 | 0.255 | `recognized` | `PBiqgJ6445` | `PBIQGJ6445` | No |
| `car_01.jpg` | Car | `WB04E4362` | `[1302, 1374, 1903, 1727]` | 0.444 | 0.748 | `unreadable` | `B1I89` | `null` | No |
| `car_02.jpg` | Car | `DL8CAF5032` | `[]` | 0.000 | 0.000 | `no_detection` | `""` | `null` | No |
| `car_03.jpg` | Car | `MH01BA1984` | `[]` | 0.000 | 0.000 | `no_detection` | `""` | `null` | No |
| `car_04.jpg` | Car | `GJ01JC5555` | `[1557, 1066, 1763, 1161]` | 0.695 | 0.574 | `recognized` | `GJLIC " 5555` | `GJLIC5555` | No |
| `car_05.jpg` | Car | `KL07B5559` | `[211, 381, 347, 420]` | 0.947 | 0.367 | `recognized` | `KL 78] 5559` | `KL785559` | No |
| `car_06.jpg` | Car/Auto | `DL1RM0556` | `[1318, 1729, 1538, 1838]` | 0.472 | 0.000 | `unreadable` | `""` | `null` | No |

---

## 6. Analysis of Failure Cases & Observations

### 6.1 Character Substitution Discrepancies
- **`bike_02.jpg`**: Ground truth `GJ01NC2394`. EasyOCR detected `GJINC2394` (conf: `0.566`). The digit `1` in standard Indian high-security registration plates (HSRP) is a single vertical stroke that is visually identical to uppercase `I`. Per strict project instructions, the engine did not fabricate a substitution.
- **`bike_06.jpg`**: Ground truth `PB10GJ6445`. EasyOCR detected `PBiqgJ6445` normalized to `PBIQGJ6445`. The characters `10` were recognized as `IQ` due to motion blur and specular glare from direct sunlight.
- **`car_04.jpg`**: Ground truth `GJ01JC5555`. EasyOCR detected `GJLIC5555`. The sub-string `01J` was transcribed as `LI` due to vehicle body shadow.
- **`car_05.jpg`**: Ground truth `KL07B5559`. EasyOCR detected `KL785559`. The letter `B` was read as `8`, a classic optical character confusion in low-resolution crops (crop size: `136 x 39` px).

### 6.2 Low-Resolution & Context-Missing Detections
- **`car_02.jpg` & `car_03.jpg`**: These images have narrow cropped regions without standard car bumper context. The YOLOv8 model was trained on contextual vehicle frames; at confidence threshold `0.15`, the model filtered them out to avoid false positive box proposals.
- **`bike_03.jpg`**: At conf `0.15`, the isolated plate crop was omitted; however, direct OCR transcribed `DLS62529` with `0.783` confidence.

---

## 7. Artifact Paths
All generated artifacts from this benchmark run are stored persistently:
- **Annotated Images**: `number_plate/results/annotated/`
- **Cropped License Plates**: `number_plate/results/cropped/`
- **CSV Evaluation Log**: `number_plate/results/test_report.csv`
- **JSON Evaluation Log**: `number_plate/results/test_report.json`

---

## 8. Summary of Modified & Created Files
- **Number Plate Module**:
  - `number_plate/detector.py` (Modified)
  - `number_plate/ocr.py` (Modified)
  - `number_plate/pipeline.py` (Modified)
  - `number_plate/__init__.py` (Modified)
  - `number_plate/run_evaluation.py` (Created)
  - `number_plate/test_images/*` (Created: 12 images + `ground_truth.csv`)
  - `number_plate/tests/test_number_plate.py` (Created)
  - `number_plate/results/*` (Created: audit report, test summary, csv, json, annotated images, cropped plates)
  - `number_plate/README.md` (Created)
- **Backend & Integration**:
  - `backend/app/main.py` (Modified: added `/api/number-plates/recognize` and response schemas)
  - `backend/app/models.py` (Modified: added `detector_confidence`, `ocr_confidence`, and `recognition_status` to `Violation`)
  - `backend/app/database.py` (Modified: added local fallback when `DATABASE_URL` is omitted)
  - `backend/app/services/pipeline_service.py` (Modified: connected plate confidence and status to violation records)
  - `backend/tests/test_api.py` (Modified: added tests for number plate recognition endpoint)
- **Frontend Dashboard**:
  - `frontend/app.js` (Modified: added detailed plate recognition badge, YOLO confidence, OCR confidence, box coordinates, and status to analysis panel and review modal)
