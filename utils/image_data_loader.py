import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_skin_disease_labels():
    """
    Return the labels for skin disease classification.
    
    Returns:
        list: List of skin disease labels
    """
    return [
        'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen's disease)',
        'Basal Cell Carcinoma',
        'Benign Keratosis-like Lesions',
        'Dermatofibroma',
        'Melanoma',
        'Melanocytic Nevi',
        'Vascular Lesions'
    ]

def prepare_ham10000_dataset(base_dir, img_size=(224, 224), batch_size=32):
    """
    Prepare the HAM10000 dataset for training and validation.
    
    Args:
        base_dir: Directory containing the HAM10000 dataset
        img_size: Target image size
        batch_size: Batch size for training
        
    Returns:
        train_ds, val_ds: TensorFlow dataset objects for training and validation
    """
    # Path structure expected:
    # base_dir/
    #   - images/    (containing all image files)
    #   - metadata/  (containing HAM10000_metadata.csv)
    
    # Load metadata
    metadata_path = os.path.join(base_dir, 'metadata', 'HAM10000_metadata.csv')
    
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}")
        # For debugging purposes, create a minimal set
        return create_debug_dataset(img_size, batch_size)
    
    df = pd.read_csv(metadata_path)
    
    # Map diagnostic categories to numerical labels
    diagnosis_mapping = {
        'akiec': 0,  # Actinic Keratoses
        'bcc': 1,    # Basal Cell Carcinoma
        'bkl': 2,    # Benign Keratosis-like Lesions
        'df': 3,     # Dermatofibroma
        'mel': 4,    # Melanoma
        'nv': 5,     # Melanocytic Nevi
        'vasc': 6    # Vascular Lesions
    }
    
    df['label'] = df['dx'].map(diagnosis_mapping)
    
    # Create image paths
    images_dir = os.path.join(base_dir, 'images')
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
    
    # Check if images exist
    df = df[df['image_path'].apply(os.path.exists)]
    
    if len(df) == 0:
        print("No valid images found")
        return create_debug_dataset(img_size, batch_size)
    
    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # Create TensorFlow datasets
    train_ds = create_dataset_from_dataframe(train_df, img_size, batch_size, augment=True)
    val_ds = create_dataset_from_dataframe(val_df, img_size, batch_size, augment=False)
    
    return train_ds, val_ds

def create_dataset_from_dataframe(df, img_size, batch_size, augment=False):
    """
    Create a TensorFlow dataset from a DataFrame.
    
    Args:
        df: DataFrame with image_path and label columns
        img_size: Target image size
        batch_size: Batch size
        augment: Whether to apply data augmentation
        
    Returns:
        TensorFlow dataset
    """
    # Create a dataset of paths and labels
    paths = df['image_path'].values
    labels = df['label'].values
    num_classes = len(np.unique(labels))
    
    # Convert to one-hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_one_hot))
    
    # Map function to load and preprocess images
    def load_and_preprocess_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation if specified
    if augment:
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        def apply_augmentation(image, label):
            image = augmentation_layers(image, training=True)
            return image, label
        
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_debug_dataset(img_size, batch_size):
    """
    Create a small synthetic dataset for debugging purposes.
    
    Args:
        img_size: Target image size
        batch_size: Batch size
        
    Returns:
        train_ds, val_ds: Small TensorFlow datasets for debugging
    """
    # Create random images and labels for debugging
    num_samples = 100
    num_classes = 7
    
    # Random images
    images = np.random.rand(num_samples, img_size[0], img_size[1], 3).astype(np.float32)
    
    # Random labels
    labels = np.random.randint(0, num_classes, size=num_samples)
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Create datasets
    full_ds = tf.data.Dataset.from_tensor_slices((images, labels_one_hot))
    train_size = int(0.8 * num_samples)
    
    train_ds = full_ds.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = full_ds.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print("Using debug dataset (synthetic data)")
    
    return train_ds, val_ds

def load_and_preprocess_image_for_prediction(image_bytes, img_size=(224, 224)):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_bytes: Image as bytes
        img_size: Target image size
        
    Returns:
        Preprocessed image tensor
    """
    # Decode image
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    
    # Resize and normalize
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    
    return img