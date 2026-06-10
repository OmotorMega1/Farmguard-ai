# FarmGuard AI — Prediction Script
# Loads the trained CNN model and predicts crop disease
# from a leaf image uploaded by a farmer

import inspect
import os
import keras

import keras.src.ops.operation as _keras_op
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# CONFIGURATION
MODEL_PATH = "models/disease_model.h5"
IMAGE_SIZE = (224, 224)

# Class names in alphabetical order — how TensorFlow assigns them
CLASS_NAMES = [
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Healthy",
]

# Healthy classes — used to determine message type
HEALTHY_CLASSES = ["Pepper Healthy", "Potato Healthy", "Tomato Healthy"]

# LOAD MODEL
# Patch Keras from_config to tolerate unknown kwargs from legacy .h5 models
# _orig_from_config = _keras_op.Operation.from_config.__func__
_orig_from_config = keras.layers.BatchNormalization.from_config

@classmethod
def _compat_from_config(cls, config):
    try:
        unwanted_keys = ['renorm', 'renorm_clipping', 'renorm_momentum']
        for key in unwanted_keys:
            config.pop(key, None)
        return _orig_from_config(config)
        # return _orig_from_config(cls, config)
    except (TypeError, ValueError):
        sig = inspect.signature(cls.__init__)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_kw:
            valid = {k for k in sig.parameters if k != 'self'}
            config = {k: v for k, v in config.items() if k in valid}
        return cls(**config)

# _keras_op.Operation.from_config = _compat_from_config
keras.layers.BatchNormalization.from_config = _compat_from_config

print("Loading disease detection model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


# SEVERITY MAPPING
# Derives severity from model confidence score
def get_severity(confidence):
    """Convert confidence score to severity level."""
    if confidence >= 0.90:
        return "Severe"
    elif confidence >= 0.70:
        return "Moderate"
    else:
        return "Mild"


# IMAGE PREPROCESSING
# Prepares a leaf image for the CNN model
def preprocess_image(image_path):
    """
    Load and preprocess a leaf image for prediction.

    Args:
        image_path (str): Path to the leaf image file

    Returns:
        numpy array: Preprocessed image ready for model input
    """
    # Load image and convert to RGB
    img = Image.open(image_path).convert("RGB")

    # Resize to 224x224 — MobileNetV2 input size
    img = img.resize(IMAGE_SIZE)

    # Normalize pixels from 0-255 to 0-1
    img_array = np.array(img) / 255.0

    # Add batch dimension — model expects (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# DISEASE PREDICTION
# Runs the preprocessed image through the CNN model
def predict_disease(image_path):
    """
    Predict crop disease from a leaf image.

    Args:
        image_path (str): Path to the leaf image

    Returns:
        dict: disease name, confidence, severity, is_healthy
    """
    # Preprocess image
    img_array = preprocess_image(image_path)

    # Run through CNN model — returns 6 probabilities
    predictions = model.predict(img_array, verbose=0)

    # Get index of highest probability
    predicted_index = np.argmax(predictions[0])

    # Get confidence score
    confidence = float(predictions[0][predicted_index])

    # Get disease name
    disease = CLASS_NAMES[predicted_index]

    # Get severity
    severity = get_severity(confidence)

    # Check if healthy
    is_healthy = disease in HEALTHY_CLASSES

    return {
        "disease": disease,
        "confidence": confidence,
        "confidence_percent": f"{confidence * 100:.2f}%",
        "severity": severity,
        "is_healthy": is_healthy,
    }


# GROQ AI TREATMENT GENERATION
# Calls Groq API to generate treatment recommendation
# or encouragement message based on prediction result
def generate_treatment(disease, confidence, severity, is_healthy):
    """
    Use Groq AI to generate a treatment recommendation
    or encouragement message.

    Args:
        disease (str): Predicted disease class name
        confidence (float): Model confidence score
        severity (str): Severity level (Mild/Moderate/Severe)
        is_healthy (bool): Whether the crop is healthy

    Returns:
        str: AI generated recommendation or message
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)

        if is_healthy:
            prompt = f"""
            You are an agricultural expert assistant helping Nigerian farmers.

            A farmer submitted a leaf photo and the AI detected:
            Status: {disease}
            Confidence: {confidence * 100:.1f}%

            Generate a warm encouraging message for the farmer.
            Include 3-4 practical tips to maintain their healthy crop.
            Keep language simple and friendly.
            Maximum 150 words.
            """
        else:
            prompt = f"""
            You are an agricultural expert assistant helping Nigerian farmers.

            A farmer submitted a leaf photo and the AI detected:
            Disease: {disease}
            Confidence: {confidence * 100:.1f}%
            Severity: {severity}

            Provide a clear actionable treatment recommendation including:
            1. What this disease is (1 sentence)
            2. Immediate steps the farmer should take (3-4 steps)
            3. Prevention tips for the future (2-3 tips)
            4. A warning about yield loss if untreated

            Keep language simple — the farmer may have limited education.
            Be direct and practical. Maximum 200 words.
            """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Groq API error: {e}")
        if is_healthy:
            return f"Your {disease.replace('Healthy', '').strip()} crop looks healthy! Keep up the great work and continue your current care routine."
        else:
            return f"Disease detected: {disease} (Severity: {severity}). Please consult your local agricultural extension officer for treatment advice."


# FULL PREDICTION PIPELINE
# Combines CNN prediction + Groq AI recommendation
# This is the main function called by the API
def analyze_leaf(image_path):
    """
    Complete leaf analysis pipeline.

    Args:
        image_path (str): Path to the uploaded leaf image

    Returns:
        dict: Complete analysis results including AI recommendation
    """
    # Step 1 — Run CNN disease prediction
    prediction = predict_disease(image_path)

    # Step 2 — Generate AI treatment recommendation
    recommendation = generate_treatment(
        disease=prediction["disease"],
        confidence=prediction["confidence"],
        severity=prediction["severity"],
        is_healthy=prediction["is_healthy"],
    )

    # Step 3 — Combine and return full results
    return {
        "disease": prediction["disease"],
        "confidence": prediction["confidence_percent"],
        "severity": prediction["severity"] if not prediction["is_healthy"] else "N/A",
        "is_healthy": prediction["is_healthy"],
        "recommendation": recommendation,
    }


# QUICK TEST
if __name__ == "__main__":
    test_folder = "data/split/test/Tomato_Early_blight"
    test_images = os.listdir(test_folder)
    test_image_path = os.path.join(test_folder, test_images[0])

    print(f"\nTesting with: {test_images[0]}")
    print("=" * 50)

    result = analyze_leaf(test_image_path)

    print(f"Disease:        {result['disease']}")
    print(f"Confidence:     {result['confidence']}")
    print(f"Severity:       {result['severity']}")
    print(f"Is Healthy:     {result['is_healthy']}")
    print("\nAI Recommendation:")
    print("-" * 50)
    print(result["recommendation"])
