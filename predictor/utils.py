import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import os

def preprocess_image(image_path):
    """Preprocess an image for model input."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read image at {image_path}. Skipping.")
        return None

    # Extract green channel (retinal images are best represented in green)
    img_green = img[:, :, 1]

    # Resize the image to 224x224
    img_resized = cv2.resize(img_green, (224, 224))

    # Apply Gaussian blur to reduce noise
    img_smoothed = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_smoothed)

    # Create a circular mask to remove background noise
    mask = np.zeros(img_clahe.shape[:2], dtype=np.uint8)
    center = (img_clahe.shape[1] // 2, img_clahe.shape[0] // 2)
    radius = min(center[0], center[1])
    cv2.circle(mask, center, radius, 255, -1)
    img_masked = cv2.bitwise_and(img_clahe, img_clahe, mask=mask)

    # Apply gamma correction for brightness adjustment
    gamma = 1.2
    img_gamma = np.power(img_masked / 255.0, gamma) * 255.0
    img_gamma = img_gamma.astype(np.uint8)

    # Normalize pixel values to [0, 1]
    img_normalized = img_gamma.astype('float32') / 255.0

    # Stack the single channel to create a 3-channel image
    img_final = np.stack([img_normalized] * 3, axis=-1)

    return img_final

def predict_image(image_path):
    """Predict DR from an image."""
    # Load the model
    model = load_model('predictor/models/ensemble_se_attention_model.h5')

    # Preprocess the image
    img = preprocess_image(image_path)
    if img is None:
        return "Error: Unable to process the image."

    # Reshape the image for model input
    img = np.expand_dims(img, axis=0)

    # Predict
    metadata = np.array([[30, 1]])  # Placeholder metadata
    prediction = model.predict({'image_input': img, 'metadata_input': metadata})

    # Return the result
    if prediction > 0.5:
        return "DR Detected: Diabetic Retinopathy is present."
    else:
        return "No DR Detected: No signs of Diabetic Retinopathy."

def predict_severity(image_path):
    """Predict severity of DR from an image."""
    # Load the severity model
    model = load_model('predictor/models/ensemble_attention_model_severity.h5')

    # Preprocess the image
    img = preprocess_image(image_path)
    if img is None:
        return "Error: Unable to process the image."

    # Reshape the image for model input
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    severity_class = np.argmax(prediction, axis=1)

    # Map class index to severity label
    severity_labels = ['Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    return f"Severity: {severity_labels[severity_class[0]]}"