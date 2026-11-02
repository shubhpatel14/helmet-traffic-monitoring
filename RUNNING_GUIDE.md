# Project Execution & Running Guide (Windows PowerShell)

This guide provides the exact commands to set up, configure, test, and run the **Autonomous Helmet Detection and Traffic Compliance Monitoring System** on Windows.

---

## 1. Prerequisites & Environment Setup

Open **Windows PowerShell** in the project directory:
```powershell
cd "c:\Users\WIN 10\Desktop\Minor Pr"
```

### Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(Verify active Python path)*:
```powershell
Get-Command python | Select-Object Source
```
*Expected: `C:\Users\WIN 10\Desktop\Minor Pr\venv\Scripts\python.exe`*

---

## 2. Dependency Verification & Installation

Verify and install all backend and computer vision requirements:
```powershell
pip install -r backend/requirements.txt
```

Verify key packages:
```powershell
python -c "import torch, cv2, ultralytics, easyocr, fastapi, sqlalchemy; print('All core libraries verified successfully!')"
```

---

## 3. Database Configuration

The application supports both **SQLite** (default zero-configuration for immediate demo) and **PostgreSQL** (production).

### Default (SQLite)
No background database server setup is needed. The database file `helmet_monitoring.db` is managed automatically.

To ensure tables exist manually before first run:
```powershell
$env:PYTHONPATH="c:\Users\WIN 10\Desktop\Minor Pr\backend"
python backend/app/init_db.py
```
*Expected output: `Database tables created successfully.`*

Verify insertion and retrieval:
```powershell
python backend/app/test_db.py
```

### Optional: PostgreSQL Production Database
1. Ensure the PostgreSQL service is running:
   ```powershell
   Get-Service -Name "*postgres*"
   ```
2. Set the `DATABASE_URL` environment variable:
   ```powershell
   $env:DATABASE_URL="postgresql://postgres:your_password@localhost:5432/helmet_monitoring"
   ```
3. Initialize tables:
   ```powershell
   python backend/app/init_db.py
   ```

---

## 4. Starting the Application

### Terminal 1: Start Backend Server
Run the FastAPI server with Uvicorn:
```powershell
cd "c:\Users\WIN 10\Desktop\Minor Pr\backend"
..\venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Server Output:*
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 5. Accessing the User Interfaces

### Operations Dashboard
Open your web browser (Chrome, Edge, Firefox) and navigate to:
```text
http://127.0.0.1:8000/dashboard/
```
Or simply:
```text
http://127.0.0.1:8000/
```
*(Root `/` automatically redirects to `/dashboard/`)*.

### Interactive API Documentation (Swagger UI)
Navigate to:
```text
http://127.0.0.1:8000/docs
```

---

## 6. Running Automated Tests

Run the full automated test suite (25/25 unit & integration tests) in a new terminal:
```powershell
cd "c:\Users\WIN 10\Desktop\Minor Pr"
.\venv\Scripts\pytest -v
```
*Expected Result: `25 passed, 10 warnings in ~15-20s`*

To run specific test modules:
```powershell
# Backend API tests
.\venv\Scripts\pytest backend/tests/test_api.py -v

# Detection and Pipeline integration tests
.\venv\Scripts\pytest backend/tests/test_pipeline.py -v

# YOLOv8 and EasyOCR number plate tests
.\venv\Scripts\pytest number_plate/tests/test_number_plate.py -v
```

---

## 7. Faculty Demonstration Procedure

1. **Open Dashboard**: Go to `http://127.0.0.1:8000/dashboard/`.
2. **Review KPI Stats**: Point out live metrics: Total Detections, Violations Flagged, Helmet Compliant, Plates Recognized, and Unreadable Plates.
3. **Inspect Audit Log**: Review the Violation Records & Audit Log table. Click on an evidence thumbnail to display the evidence modal with bounding boxes, YOLO detector confidence, and OCR confidence.
4. **Demonstrate Approval Action**: Click **Approve** or **Reject** on a record and observe the immediate real-time badge and KPI counter update.
5. **Switch to Studio**: Click **Frame Analysis Studio (YOLOv8 + OCR)** tab.
6. **Upload Motorcycle**: Click **Select Local Image** and select `number_plate/test_images/bike_01.jpg`.
   - Observe real-time detection: `With Helmet` (Confidence: 0.454), green bounding box.
   - Point out that record is saved to the database.
7. **Upload Vehicle with Number Plate**: Select `number_plate/test_images/bike_02.jpg` or `number_plate/test_images/car_04.jpg`.
   - Observe plate localization (`[571, 612, 1142, 768]`), YOLO confidence (`0.825`), and EasyOCR text extraction (`GJINC2394`, OCR Confidence `0.566`).
8. **Verify Persistence in Audit Log**: Switch back to Tab 1 to demonstrate that the newly uploaded frames now appear at the top of the database audit table.
