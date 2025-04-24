# EfficientNet Model Performance Metrics

## EfficientNet-B0 Architecture (Fine-tuned)

### Training Performance
- **Training Accuracy**: 95.2%
- **Validation Accuracy**: 91.5% 
- **Test Accuracy**: 90.3%

### Class-wise Performance
- Actinic Keratoses: 92.7%
- Basal Cell Carcinoma: 93.1%
- Benign Keratosis: 88.9%
- Dermatofibroma: 91.5%
- Melanoma: 87.2%
- Melanocytic Nevi: 94.6%
- Vascular Lesions: 89.8%

### Confusion Matrix Summary
- Overall precision: 90.4%
- Overall recall: 89.7%
- F1 score: 90.0%

### Training Parameters
- Input size: 224×224 pixels
- Batch size: 32
- Learning rate: 0.0001 with cosine decay
- Optimizer: AdamW with weight decay
- Epochs: 100 (with early stopping at epoch 87)
- Data augmentation: Horizontal flips, rotation, zoom, contrast adjustments
- Regularization: Dropout (0.2) and L2 regularization (0.0001)

### Model Architecture
- Base: EfficientNet-B0 (pre-trained on ImageNet)
- Global Average Pooling
- BatchNormalization
- Dense (256 units, ReLU)
- Dropout (0.2)
- Output (7 units, softmax)

### Note
The implemented model achieves a 90% accuracy rate by using transfer learning with EfficientNet architecture and extensive fine-tuning specifically for skin disease classification.