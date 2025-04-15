import numpy as np
from PIL import Image
import io
import random
from utils.data_loader import get_skin_disease_labels

def preprocess_image(image_bytes):
    """
    Preprocess image for analysis.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        image: Processed PIL image
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize image to standard size
    image = image.resize((224, 224))
    
    return image

def get_image_prediction(image_bytes):
    """
    Predict skin disease based on image.
    This is a simplified version that returns random predictions
    for demonstration purposes.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        tuple: (predicted_disease, confidence_percentage)
    """
    try:
        # Process the image
        image = preprocess_image(image_bytes)
        
        # Get skin disease labels
        disease_labels = get_skin_disease_labels()
        
        # For demonstration, randomly select a disease and confidence
        # In a real application, this would use a trained model
        predicted_class_index = random.randint(0, len(disease_labels) - 1)
        confidence = random.uniform(65.0, 95.0)
        
        # Get the predicted disease name
        predicted_disease = disease_labels[predicted_class_index]
        
        return predicted_disease, confidence
    
    except Exception as e:
        print(f"Error in image prediction: {e}")
        
        # Return a fallback prediction
        return "Melanocytic nevi", 65.0
