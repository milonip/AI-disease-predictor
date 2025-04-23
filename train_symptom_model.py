import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import argparse

def load_kaggle_disease_dataset(dataset_path):
    """
    Load the Kaggle disease prediction dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        X_train, X_test, y_train, y_test: Training and testing data splits
    """
    print(f"Loading disease dataset from: {dataset_path}")
    
    # Expected file paths
    training_file = os.path.join(dataset_path, "Training.csv")
    testing_file = os.path.join(dataset_path, "Testing.csv")
    
    # Check if files exist
    if not os.path.exists(training_file) or not os.path.exists(testing_file):
        raise FileNotFoundError(f"Dataset files not found at {dataset_path}")
    
    # Load the data
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(testing_file)
    
    # Display information about the dataset
    print(f"Training set shape: {train_data.shape}")
    print(f"Testing set shape: {test_data.shape}")
    print(f"Number of classes (diseases): {train_data['prognosis'].nunique()}")
    print(f"Sample symptoms: {list(train_data.columns[:-1])[:5]}")
    
    # Separate features and target
    X_train = train_data.drop('prognosis', axis=1)
    y_train = train_data['prognosis']
    
    X_test = test_data.drop('prognosis', axis=1)
    y_test = test_data['prognosis']
    
    return X_train, X_test, y_train, y_test

def train_random_forest_model(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train a Random Forest classifier for disease prediction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        
    Returns:
        Trained Random Forest model
    """
    print("Training Random Forest model...")
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        accuracy: Model accuracy
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix Shape:", cm.shape)
    
    return accuracy

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the trained model.
    
    Args:
        model: Trained model
        feature_names: Names of the features
        
    Returns:
        DataFrame of features sorted by importance
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance in descending order
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nTop 10 Important Symptoms:")
    print(feature_importance.head(10))
    
    return feature_importance

def main(dataset_path, model_save_path="models", n_estimators=100, max_depth=None):
    """
    Main function to train and evaluate the disease prediction model.
    
    Args:
        dataset_path: Path to the Kaggle disease dataset
        model_save_path: Path to save the trained model
        n_estimators: Number of trees in the Random Forest
        max_depth: Maximum depth of the trees
    """
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Load dataset
    X_train, X_test, y_train, y_test = load_kaggle_disease_dataset(dataset_path)
    
    # Train model
    model = train_random_forest_model(X_train, y_train, n_estimators, max_depth)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Get feature importance
    feature_importance = get_feature_importance(model, X_train.columns)
    
    # Save model and related information
    model_file = os.path.join(model_save_path, "symptom_disease_model.joblib")
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    # Save model information
    with open(os.path.join(model_save_path, "symptom_model_details.txt"), "w") as f:
        f.write(f"Symptom-based Disease Prediction Model\n")
        f.write(f"====================================\n")
        f.write(f"Model type: Random Forest\n")
        f.write(f"Number of estimators: {n_estimators}\n")
        f.write(f"Maximum depth: {max_depth if max_depth is not None else 'None (unlimited)'}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"\nNumber of features (symptoms): {X_train.shape[1]}\n")
        f.write(f"Number of classes (diseases): {len(y_train.unique())}\n")
        f.write(f"\nTop 10 Important Symptoms:\n")
        for index, row in feature_importance.head(10).iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    
    # Save feature importance to CSV
    feature_importance.to_csv(os.path.join(model_save_path, "symptom_importance.csv"), index=False)
    
    print(f"Model details saved to {os.path.join(model_save_path, 'symptom_model_details.txt')}")
    print(f"Feature importance saved to {os.path.join(model_save_path, 'symptom_importance.csv')}")
    
    return model, accuracy

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train disease prediction model based on symptoms")
    parser.add_argument("--dataset_path", type=str, default="./data/disease_prediction",
                        help="Path to the Kaggle disease dataset")
    parser.add_argument("--model_save_path", type=str, default="./models",
                        help="Directory to save models")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the Random Forest")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth of the trees")
    
    args = parser.parse_args()
    
    # Train and evaluate model
    main(
        dataset_path=args.dataset_path,
        model_save_path=args.model_save_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )