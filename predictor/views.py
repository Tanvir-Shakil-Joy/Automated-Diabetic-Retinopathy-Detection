from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from .utils import predict_image, predict_severity
import os

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        # Save the uploaded image
        image_file = request.FILES['image']
        file_name = default_storage.save(image_file.name, ContentFile(image_file.read()))
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        # Predict the image
        result = predict_image(file_path)

        # Pass the image path to the template
        return render(request, 'predictor/home.html', {'result': result, 'image_path': file_path})

    return render(request, 'predictor/home.html')

def severity_check(request):
    if request.method == 'POST':
        # Get the image path from the form
        image_path = request.POST.get('image_path')

        # Predict the severity
        result = predict_severity(image_path)

        # Delete the uploaded image after prediction
        os.remove(image_path)

        return render(request, 'predictor/severity_check.html', {'result': result})

    return render(request, 'predictor/severity_check.html')

def no_dr(request):
    return render(request, 'predictor/no_dr.html')

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import numpy as np
import cv2
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

    # Add batch dimension
    img_final = np.expand_dims(img_final, axis=0)

    return img_final

def predict(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            # Save the uploaded image
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(f'uploads/{image_file.name}', image_file)
            image_url = fs.url(filename)
            image_path = os.path.join('media', filename)
            
            # Ensure the uploads directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Preprocess the image
            img_array = preprocess_image(image_path)
            
            if img_array is None:
                raise ValueError("Image preprocessing failed")
            
            # Load models with proper compilation
            dr_detection_model = tf.keras.models.load_model('predictor/model/ensemble_se_attention_model.h5', compile=False)
            severity_model = tf.keras.models.load_model('predictor/model/ensemble_attention_model_severity.h5', compile=False)
            
            # Compile severity model with proper configuration
            severity_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Create dummy metadata input for binary classification
            metadata_input = np.zeros((1, 2))
            
            # First prediction: DR or No DR with both inputs
            dr_prediction = dr_detection_model.predict({'image_input': img_array, 'metadata_input': metadata_input}, verbose=0)
            dr_prob = dr_prediction[0][0]
            # Fix: Change the threshold logic - if probability > 0.5, it means DR is present
            has_dr = 1 if dr_prob > 0.5 else 0
            dr_confidence = float(dr_prob if has_dr else 1 - dr_prob) * 100
            
            context = {
                'image_url': image_url,
                'has_dr': has_dr,
                'dr_confidence': f'{dr_confidence:.2f}'
            }
            
            # If DR is detected, predict severity with single input
            if has_dr:
                # Create input tensor with correct shape
                severity_input = tf.convert_to_tensor(img_array, dtype=tf.float32)
                
                # Get model predictions
                severity_prediction = severity_model(severity_input, training=False)
                severity_classes = ['Mild', 'Moderate', 'Severe', 'Proliferative_DR']
                
                # Get raw logits and normalize them
                logits = severity_prediction.numpy()[0]
                # Normalize logits to prevent fixed predictions
                normalized_logits = logits - np.mean(logits)
                
                # Convert to probabilities using softmax with temperature
                temperature = 1.0
                severity_probs = tf.nn.softmax(normalized_logits / temperature).numpy()
                
                # Get class with highest probability
                severity_class_idx = np.argmax(severity_probs)
                severity_class = severity_classes[severity_class_idx]
                severity_confidence = float(severity_probs[severity_class_idx]) * 100
                
                # Store probabilities
                class_probabilities = {
                    cls: float(prob) * 100 
                    for cls, prob in zip(severity_classes, severity_probs)
                }
                
                # Debug logging
                print("Raw logits:", logits)
                print("Normalized logits:", normalized_logits)
                print("Probabilities after softmax:")
                for cls, prob in class_probabilities.items():
                    print(f"{cls}: {prob:.4f}%")
                
                context.update({
                    'severity': {
                        'class': severity_class,
                        'confidence': f'{severity_confidence:.2f}',
                        'all_probabilities': class_probabilities
                    }
                })
                
                # Debug logging
                print("Raw prediction:", severity_prediction)
                print("Probabilities after softmax:")
                for cls, prob in class_probabilities.items():
                    print(f"{cls}: {prob:.2f}%")
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            context = {
                'error_message': f'Error during prediction: {str(e)}'
            }
        
        return render(request, 'predictor/result.html', context)
    
    return render(request, 'predictor/predict.html')