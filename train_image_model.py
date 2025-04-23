import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import argparse
from utils.image_data_loader import prepare_ham10000_dataset, get_skin_disease_labels

def train_skin_disease_model(dataset_path, epochs=20, batch_size=32, model_save_path="models"):
    """
    Train a model for skin disease classification using HAM10000 dataset.
    
    Args:
        dataset_path: Path to the HAM10000 dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_save_path: Directory to save models
        
    Returns:
        Trained model and training history
    """
    print(f"Training skin disease model using data from: {dataset_path}")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Prepare datasets
    train_ds, val_ds = prepare_ham10000_dataset(dataset_path, batch_size=batch_size)
    
    # Debug Dataset
    def inspect_dataset(dataset, name="Dataset"):
        print(f"\nInspecting {name}:")
        for images, labels in dataset.take(1):
            print("Image Shape:", images.shape)
            print("Image Min/Max:", tf.reduce_min(images).numpy(), tf.reduce_max(images).numpy())
            print("Label Shape:", labels.shape)
            print("Sample Labels:", labels[:5].numpy())
        return dataset

    train_ds = inspect_dataset(train_ds, "train_ds")
    val_ds = inspect_dataset(val_ds, "val_ds")

    # Class Weights
    try:
        train_labels = [np.argmax(label.numpy()) for _, label in train_ds.unbatch()]
        train_labels = np.array(train_labels)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weight_dict = dict(enumerate(class_weights))
        print("Class Weights:", class_weight_dict)
    except Exception as e:
        print(f"Error computing class weights: {e}")
        class_weight_dict = None

    num_classes = 7  # HAM10000 has 7 classes

    # Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    # Model Architecture - EfficientNetB3 for a balance of performance and size
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.EfficientNetB3(include_top=False, weights="imagenet")(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    # Model Summary
    model.summary()

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_save_path, "best_skin_model.keras"), 
            save_best_only=True, 
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6
        )
    ]

    # Train
    print("Starting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(model_save_path, "skin_disease_model_final.keras")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Final evaluation
    print("Evaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
    
    # Save model details
    labels = get_skin_disease_labels()
    with open(os.path.join(model_save_path, "model_details.txt"), "w") as f:
        f.write(f"Skin Disease Model\n")
        f.write(f"=================\n")
        f.write(f"Final Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Final Validation Loss: {val_loss:.4f}\n")
        f.write(f"\nClasses:\n")
        for i, label in enumerate(labels):
            f.write(f"{i}: {label}\n")
    
    return model, history

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train skin disease classification model")
    parser.add_argument("--dataset_path", type=str, default="./data/ham10000",
                        help="Path to HAM10000 dataset")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--model_save_path", type=str, default="./models",
                        help="Directory to save models")
    
    args = parser.parse_args()
    
    # Train the model
    train_skin_disease_model(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path
    )