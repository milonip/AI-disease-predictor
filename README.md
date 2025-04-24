# AI Disease Predictor

A comprehensive disease prediction platform that combines advanced machine learning techniques for accurate medical diagnostics through dual prediction models: symptom-based and image-based analysis.

## Overview

This application provides two methods for disease prediction:

1. **Symptom-Based Prediction**: Users can select symptoms from a comprehensive list, and the system predicts the most likely disease based on those symptoms.

2. **Image-Based Skin Disease Prediction**: Users can upload images of skin conditions, and the system analyzes the image to predict the type of skin disease.

## Key Features

- **Dual Prediction Models**: Offers both symptom-based and image-based disease prediction
- **User-Friendly Interface**: Clean, intuitive UI with tabbed navigation
- **High Accuracy**: Optimized algorithms for accurate disease prediction
- **Responsive Design**: Works well on various devices and screen sizes

## Technical Details

### Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **NumPy/Pandas**: Data manipulation and analysis
- **PIL**: Image processing
- **Scikit-learn**: Machine learning algorithms for symptom-based prediction
- **Custom Image Analysis**: Specialized algorithms for skin disease classification

### Prediction Methods

#### Symptom-Based Prediction
The symptom-based model uses a Random Forest classifier trained on the Kaggle disease prediction dataset. It processes user-selected symptoms to predict the most likely disease with a confidence score.

#### Image-Based Prediction
The image-based model uses advanced color pattern recognition and texture analysis to identify skin diseases from the HAM10000 dataset categories:

1. Actinic keratoses and intraepithelial carcinoma
2. Basal cell carcinoma
3. Benign keratosis-like lesions
4. Dermatofibroma
5. Melanoma
6. Melanocytic nevi
7. Vascular lesions

## Project Structure

```
AI-disease-predictor/
├── app.py                      # Main Streamlit application
├── assets/                     # Static assets and resources
│   └── symptoms_list.py        # List of symptoms for the UI
├── models/                     # Trained models and model-related scripts
│   ├── create_test_model.py    # Script to create test models
│   └── model_performance.md    # Documentation of model performance
├── utils/                      # Utility functions and modules
│   ├── data_loader.py          # Data loading utilities
│   ├── image_data_loader.py    # Image data processing
│   ├── image_predictor.py      # Image-based disease prediction
│   ├── symptom_predictor.py    # Symptom-based disease prediction
│   └── model_trainer.py        # Model training utilities
└── README.md                   # Project documentation
```

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/milonip/AI-disease-predictor.git
   cd AI-disease-predictor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **For Symptom-Based Prediction**:
   - Select the "Symptom Based" tab
   - Check the symptoms that apply
   - Click "Predict" to see the results

2. **For Image-Based Prediction**:
   - Select the "Image Based" tab
   - Upload an image of the skin condition
   - Click "Predict" to see the analysis results

## Disclaimer

This application is for educational purposes only and should not be used for self-diagnosis. Always consult with a qualified healthcare professional for medical advice.

## Acknowledgments

- HAM10000 dataset for skin disease images
- Kaggle disease prediction dataset for symptom-based model training
- Streamlit for the web application framework

## License

This project is licensed under the MIT License - see the LICENSE file for details.