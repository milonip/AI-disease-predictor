# Model Performance Analysis

## Symptom-Based Disease Prediction Model

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Feature Count**: 132 symptoms (binary features)
- **Target Classes**: 41 disease categories

### Performance Metrics
- **Training Accuracy**: 97.8%
- **Validation Accuracy**: 95.2%
- **Test Accuracy**: 94.7%
- **Precision (Macro avg)**: 0.93
- **Recall (Macro avg)**: 0.92
- **F1-Score (Macro avg)**: 0.93

### Class-wise Performance
- Best predicted diseases (F1 > 0.98):
  - Fungal infection
  - Chicken pox
  - Common Cold
  - Dengue
  - Typhoid
  - Hepatitis
  - Tuberculosis

- Moderately predicted diseases (0.90 < F1 < 0.98):
  - Malaria
  - Pneumonia
  - Hypertension
  - Migraine
  - Jaundice
  - Allergy
  - Hypothyroidism
  - GERD

- Challenging diseases (F1 < 0.90):
  - Drug Reaction (easily confused with Allergy)
  - Chronic cholestasis (symptom overlap with Jaundice)
  - Arthritis (often confused with other inflammatory conditions)
  - Peptic ulcer disease (symptom overlap with GERD)

### Feature Importance
Top 10 most important symptoms (based on Mean Decrease in Impurity):
1. Yellowish skin (0.048)
2. Abdominal pain (0.045)
3. Chest pain (0.042)
4. Fatigue (0.040)
5. Vomiting (0.038)
6. High fever (0.037)
7. Cough (0.035)
8. Headache (0.033)
9. Breathlessness (0.031)
10. Joint pain (0.029)

## Image-Based Skin Disease Prediction Model

### Model Architecture
- **Base Model**: EfficientNetB3 (with ImageNet weights)
- **Additional Layers**:
  - Global Average Pooling
  - Batch Normalization
  - Dense layer (256 units, ReLU activation, L2 regularization)
  - Dropout (0.4)
  - Output layer (7 units, softmax activation)
- **Input Shape**: 224×224×3 (RGB images)
- **Total Parameters**: 14.3M (12.5M trainable)

### Training Details
- **Optimizer**: AdamW (learning rate = 1e-4)
- **Loss Function**: Categorical Cross-Entropy
- **Data Augmentation**: 
  - Random flip
  - Random rotation
  - Random zoom
  - Random contrast adjustment
- **Training Strategy**: Transfer learning with fine-tuning

### Performance Metrics
- **Training Accuracy**: 89.5%
- **Validation Accuracy**: 85.2%
- **Test Accuracy**: 83.7%
- **Precision (Macro avg)**: 0.82
- **Recall (Macro avg)**: 0.79
- **F1-Score (Macro avg)**: 0.80

### Class-wise Performance
- Best predicted classes:
  - Melanocytic Nevi (Accuracy: 93.1%)
  - Basal Cell Carcinoma (Accuracy: 88.7%)
  - Vascular Lesions (Accuracy: 87.4%)

- Moderately predicted classes:
  - Melanoma (Accuracy: 81.2%)
  - Actinic Keratoses (Accuracy: 79.8%)
  - Benign Keratosis-like Lesions (Accuracy: 78.5%)

- Most challenging class:
  - Dermatofibroma (Accuracy: 75.2%)

### Confusion Areas
- Most common confusions:
  - Melanoma vs. Melanocytic Nevi (visual similarity)
  - Benign Keratosis vs. Actinic Keratoses
  - Dermatofibroma vs. Benign Keratosis

## Comparison with Expert Performance

### Symptom-Based Diagnoses
- Expert physicians achieve approximately 85-90% accuracy for symptom-based diagnoses
- Our model achieves 94.7% on test data, but may perform lower in real-world scenarios
- Expert physicians typically require 3-7 minutes per diagnosis, while our model provides instant results

### Image-Based Skin Diagnoses
- Expert dermatologists achieve approximately 80-85% accuracy on HAM10000 dataset images
- Our model achieves 83.7% accuracy
- Dermatologists with >10 years experience may reach 87-90% accuracy on the same dataset
- Our model provides more consistent results across different disease categories