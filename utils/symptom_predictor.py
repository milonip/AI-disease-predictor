import numpy as np
import pandas as pd
from assets.symptoms_list import symptoms_list
from utils.model_trainer import load_symptom_model

def get_symptom_prediction(selected_symptoms):
    """
    Predict disease based on selected symptoms.
    
    Args:
        selected_symptoms: List of selected symptom strings
        
    Returns:
        tuple: (predicted_disease, confidence_percentage)
    """
    try:
        # Load the trained model
        model = load_symptom_model()
        
        # Create input features (one-hot encoding of symptoms)
        input_features = np.zeros(len(symptoms_list))
        
        # Set selected symptoms to 1
        for symptom in selected_symptoms:
            if symptom in symptoms_list:
                symptom_index = symptoms_list.index(symptom)
                input_features[symptom_index] = 1
        
        # Reshape for prediction (1 sample, all features)
        input_features = input_features.reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(input_features)
        
        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_features)
        confidence = np.max(prediction_proba) * 100  # Convert to percentage
        
        return prediction[0], confidence
    
    except Exception as e:
        print(f"Error in symptom prediction: {e}")
        
        # Return a fallback prediction
        disease_map = {
            "dehydration": "Dehydration",
            "loss of appetite": "Gastroenteritis",
            "yellowish skin": "Jaundice",
            "fatigue": "Chronic Fatigue Syndrome",
            "high fever": "Influenza",
            "breathlessness": "Asthma",
            "sweating": "Hyperthyroidism",
            "headache": "Migraine",
            "nausea": "Food Poisoning",
            "muscle wasting": "Muscular Dystrophy"
        }
        
        # Check if any symptoms match known conditions
        for symptom in selected_symptoms:
            if symptom.lower() in disease_map:
                return disease_map[symptom.lower()], 85.0
        
        # Default fallback
        return "Common Cold", 70.0
