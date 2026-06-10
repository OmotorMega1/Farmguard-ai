# FarmGuard AI — Test Suite
# Tests the prediction pipeline across all 6 disease classes

import os
import sys

# Add project root to path so we can import predict.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_disease, get_severity

# TEST FOLDERS
# Points to our existing test set images
TEST_FOLDERS = {
    "Pepper Bacterial Spot": "data/split/test/Pepper__bell___Bacterial_spot",
    "Pepper Healthy":        "data/split/test/Pepper__bell___healthy",
    "Potato Early Blight":   "data/split/test/Potato___Early_blight",
    "Potato Healthy":        "data/split/test/Potato___healthy",
    "Tomato Early Blight":   "data/split/test/Tomato_Early_blight",
    "Tomato Healthy":        "data/split/test/Tomato_healthy"
}

# TEST 1 — MODEL LOADING
# Confirms the CNN model loads without errors
def test_model_loading():
    print("TEST 1 — Model Loading")
    print("-" * 40)
    try:
        from src.predict import model
        assert model is not None
        print("PASSED ✅ — Model loaded successfully\n")
        return True
    except Exception as e:
        print(f"FAILED ❌ — {e}\n")
        return False

# TEST 2 — PREDICTION STRUCTURE
# Confirms prediction returns all required fields
def test_prediction_structure():
    print("TEST 2 — Prediction Structure")
    print("-" * 40)
    try:
        folder = TEST_FOLDERS["Tomato Early Blight"]
        image_path = os.path.join(folder, os.listdir(folder)[0])
        result = predict_disease(image_path)

        # Check all required fields exist
        required_fields = ["disease", "confidence", "confidence_percent", "severity", "is_healthy"]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Check confidence is between 0 and 1
        assert 0 <= result["confidence"] <= 1

        # Check severity is one of the valid options
        assert result["severity"] in ["Mild", "Moderate", "Severe"]

        # Check is_healthy is boolean
        assert isinstance(result["is_healthy"], bool)

        print("PASSED ✅ — All fields present and valid")
        print(f"  Disease:    {result['disease']}")
        print(f"  Confidence: {result['confidence_percent']}")
        print(f"  Severity:   {result['severity']}")
        print(f"  Is Healthy: {result['is_healthy']}\n")
        return True

    except Exception as e:
        print(f"FAILED ❌ — {e}\n")
        return False

# TEST 3 — ALL 6 CLASSES
# Tests 3 images per class and checks prediction accuracy
def test_all_classes():
    print("TEST 3 — All 6 Classes (3 images each)")
    print("-" * 40)

    correct = 0
    total = 0
    failed_classes = []

    for true_class, folder in TEST_FOLDERS.items():
        images = os.listdir(folder)[:3]
        class_correct = 0

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            result = predict_disease(img_path)
            if result["disease"] == true_class:
                correct += 1
                class_correct += 1
            total += 1

        status = "✅" if class_correct == 3 else "⚠️"
        print(f"  {status} {true_class}: {class_correct}/3 correct")

        if class_correct < 3:
            failed_classes.append(true_class)

    accuracy = (correct / total) * 100
    passed = accuracy >= 90

    print(f"\n  Overall: {correct}/{total} correct ({accuracy:.1f}%)")

    if passed:
        print(f"PASSED ✅ — Accuracy above 90% threshold\n")
    else:
        print(f"FAILED ❌ — Accuracy below 90% threshold")
        print(f"  Weak classes: {failed_classes}\n")

    return passed

# TEST 4 — HEALTHY CROP DETECTION
# Confirms healthy crops are correctly flagged as healthy
def test_healthy_detection():
    print("TEST 4 — Healthy Crop Detection")
    print("-" * 40)

    healthy_folders = {
        "Pepper Healthy": TEST_FOLDERS["Pepper Healthy"],
        "Potato Healthy": TEST_FOLDERS["Potato Healthy"],
        "Tomato Healthy": TEST_FOLDERS["Tomato Healthy"]
    }

    all_passed = True

    for true_class, folder in healthy_folders.items():
        image_path = os.path.join(folder, os.listdir(folder)[0])
        result = predict_disease(image_path)

        is_correct = result["is_healthy"] == True
        status = "✅" if is_correct else "❌"
        print(f"  {status} {true_class}: is_healthy = {result['is_healthy']}")

        if not is_correct:
            all_passed = False

    print(f"\n{'PASSED ✅ — All healthy crops correctly identified' if all_passed else 'FAILED ❌ — Some healthy crops misidentified'}\n")
    return all_passed

# TEST 5 — SEVERITY MAPPING
# Confirms severity levels are correctly assigned
def test_severity_mapping():
    print("TEST 5 — Severity Mapping")
    print("-" * 40)

    tests = [
        (0.95, "Severe"),
        (0.75, "Moderate"),
        (0.55, "Mild")
    ]

    all_passed = True

    for confidence, expected in tests:
        result = get_severity(confidence)
        passed = result == expected
        status = "✅" if passed else "❌"
        print(f"  {status} Confidence {confidence:.0%} → {result} (expected {expected})")
        if not passed:
            all_passed = False

    print(f"\n{'PASSED ✅ — Severity mapping correct' if all_passed else 'FAILED ❌ — Severity mapping incorrect'}\n")
    return all_passed

# RUN ALL TESTS
if __name__ == "__main__":
    print("=" * 50)
    print("FARMGUARD AI — TEST SUITE")
    print("=" * 50)
    print()

    results = {
        "Model Loading":        test_model_loading(),
        "Prediction Structure": test_prediction_structure(),
        "All 6 Classes":        test_all_classes(),
        "Healthy Detection":    test_healthy_detection(),
        "Severity Mapping":     test_severity_mapping()
    }

    print("=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status} — {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🚀 All tests passed — Ready for deployment!")
    else:
        print("\n⚠️  Fix failing tests before deploying.")