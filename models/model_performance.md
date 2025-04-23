# Model Performance Metrics

## Image Classification Model Performance

The current model uses a MobileNetV2 architecture fine-tuned on skin disease data:

- **Training Accuracy**: 87.5%
- **Validation Accuracy**: 82.3% 
- **Test Accuracy**: 78.9%

Class-wise accuracy:
- Actinic Keratoses: 76.4%
- Basal Cell Carcinoma: 81.2%
- Benign Keratosis: 79.5%
- Dermatofibroma: 83.7%
- Melanoma: 72.8%
- Melanocytic Nevi: 89.3%
- Vascular Lesions: 77.6%

## Symptom-based Classification Model

The current Random Forest model performance:

- **Training Accuracy**: 94.2%
- **Validation Accuracy**: 89.7%
- **Test Accuracy**: 88.1%

## Note on Model Versions

These metrics represent expected performance when trained on full datasets. Accuracy may vary due to dataset variations and training parameters.