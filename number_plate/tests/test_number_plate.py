import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from number_plate.detector import DEFAULT_WEIGHTS_PATH, detect_plates
from number_plate.ocr import normalize_plate_text, recognize_plate_text
from number_plate.pipeline import process_traffic_image


@pytest.fixture
def genuine_test_image():
    """Return path to a genuine test vehicle image known to contain a plate."""
    test_img = os.path.join(
        PROJECT_ROOT, "number_plate", "test_images", "bike_02.jpg"
    )
    if os.path.exists(test_img):
        return test_img
    pytest.skip("Test image bike_02.jpg not found")


@pytest.fixture
def blank_image():
    """Return a blank 640x480 RGB image containing no vehicles or plates."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_valid_vehicle_image(genuine_test_image):
    """Test full pipeline on a valid vehicle image."""
    results = process_traffic_image(genuine_test_image, conf_threshold=0.20)
    assert isinstance(results, list)
    assert len(results) >= 1

    first = results[0]
    assert "plate_box" in first
    assert "detector_confidence" in first
    assert "ocr_confidence" in first
    assert "plate_text" in first
    assert "raw_ocr_text" in first
    assert "status" in first
    assert first["status"] in ["recognized", "unreadable"]
    assert len(first["plate_box"]) == 4
    assert 0.0 <= first["detector_confidence"] <= 1.0


def test_invalid_image_type():
    """Test error handling when input is an invalid type."""
    with pytest.raises(TypeError):
        detect_plates(123456)  # integer is invalid


def test_incorrect_file_path():
    """Test that non-existent image path raises FileNotFoundError in detector."""
    with pytest.raises(FileNotFoundError):
        detect_plates("non_existent_image_path_12345.jpg")


def test_image_without_number_plate(blank_image):
    """Test that an image with no number plate returns an empty list without error."""
    results = process_traffic_image(blank_image)
    assert isinstance(results, list)
    assert len(results) == 0


def test_missing_yolo_checkpoint(blank_image):
    """Test that a non-existent YOLO checkpoint path raises FileNotFoundError."""
    bogus_path = "non_existent_weights_dir/missing_model.pt"
    with pytest.raises(FileNotFoundError):
        detect_plates(blank_image, weights_path=bogus_path)


def test_empty_crop():
    """Test that passing an empty numpy array handles gracefully."""
    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    plates = detect_plates(empty_arr)
    assert plates == []

    ocr_res = recognize_plate_text(empty_arr)
    assert ocr_res["status"] == "unreadable"
    assert ocr_res["normalized_text"] is None
    assert ocr_res["ocr_confidence"] == 0.0

    pipe_res = process_traffic_image(empty_arr)
    assert pipe_res == []


def test_ocr_unreadable_text():
    """Test that solid noise or blank crop is marked as unreadable."""
    noise_crop = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
    ocr_res = recognize_plate_text(noise_crop)
    assert ocr_res["status"] in ["unreadable", "recognized"]
    if ocr_res["status"] == "unreadable":
        assert ocr_res["normalized_text"] is None


def test_low_confidence_filtering(genuine_test_image):
    """Test that setting an impossibly high confidence threshold filters out detections."""
    results = process_traffic_image(genuine_test_image, conf_threshold=0.9999)
    assert len(results) == 0


def test_indian_plate_normalization():
    """Test the normalization function against various standard Indian plate formats."""
    # Standard format: State + District + Series + 4 digits
    norm, valid = normalize_plate_text("DL 08 CAF 5032")
    assert norm == "DL08CAF5032"
    assert valid is True

    # Standard format with hyphens
    norm, valid = normalize_plate_text("MH-12-DE-1433")
    assert norm == "MH12DE1433"
    assert valid is True

    # Plate with IND country prefix (HSRP)
    norm, valid = normalize_plate_text("IND GJ 01 AB 1234")
    assert norm == "GJ01AB1234"
    assert valid is True

    # Bharat (BH) Series
    norm, valid = normalize_plate_text("22 BH 1234 AA")
    assert norm == "22BH1234AA"
    assert valid is True

    # Fragmented or too short text should be rejected
    norm, valid = normalize_plate_text("DL 1")
    assert norm is None
    assert valid is False

    # Empty text
    norm, valid = normalize_plate_text("")
    assert norm is None
    assert valid is False


def test_pipeline_output_contract(genuine_test_image):
    """Test that pipeline output matches the exact required dictionary schema."""
    results = process_traffic_image(genuine_test_image, conf_threshold=0.20)
    for item in results:
        # Check all required keys exist
        assert "plate_text" in item
        assert "raw_ocr_text" in item
        assert "detector_confidence" in item
        assert "ocr_confidence" in item
        assert "plate_box" in item
        assert "status" in item

        # Check types
        assert item["plate_text"] is None or isinstance(item["plate_text"], str)
        assert isinstance(item["raw_ocr_text"], str)
        assert isinstance(item["detector_confidence"], float)
        assert isinstance(item["ocr_confidence"], float)
        assert isinstance(item["plate_box"], list)
        assert item["status"] in ["recognized", "unreadable"]
