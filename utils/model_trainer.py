import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

def train_symptom_model(X_train, y_train):
    """
    Train a machine learning model for symptom-based disease prediction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        model: Trained model
    """
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    return model

def evaluate_symptom_model(model, X_test, y_test):
    """
    Evaluate the symptom prediction model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        float: Accuracy score
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def save_symptom_model(model, filename="symptom_model.joblib"):
    """
    Save the trained symptom model to disk.
    
    Args:
        model: Trained model
        filename: Path to save the model
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join("models", filename))

def load_symptom_model(filename="symptom_model.joblib"):
    """
    Load the trained symptom model from disk.
    
    Args:
        filename: Path to the saved model
        
    Returns:
        model: Loaded model
    """
    try:
        # Load model
        model = joblib.load(os.path.join("models", filename))
        return model
    except:
        # If model doesn't exist, train a new one
        from utils.data_loader import load_symptom_data
        X_train, _, y_train, _ = load_symptom_data()
        model = train_symptom_model(X_train, y_train)
        save_symptom_model(model)
        return model

def build_skin_disease_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Build a CNN model for skin disease classification using transfer learning.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of disease classes
        
    Returns:
        model: Compiled model
    """
    # Use MobileNetV2 as base model (lightweight and effective)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_skin_disease_model(model, train_dataset, val_dataset, epochs=10):
    """
    Train the skin disease classification model.
    
    Args:
        model: Compiled model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Train model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset
    )
    
    return model, history

def save_skin_disease_model(model, filename="skin_disease_model"):
    """
    Save the trained skin disease model to disk.
    
    Args:
        model: Trained model
        filename: Path to save the model
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model.save(os.path.join("models", filename))

def load_skin_disease_model(filename="skin_disease_model"):
    """
    Load the trained skin disease model from disk.
    
    Args:
        filename: Path to the saved model
        
    Returns:
        model: Loaded model
    """
    try:
        # Load model
        model = tf.keras.models.load_model(os.path.join("models", filename))
        return model
    except:
        # If model doesn't exist, create a new one
        # Note: In a real application, we would train this model properly
        model = build_skin_disease_model()
        return model
