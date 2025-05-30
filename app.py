import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from PIL import Image
import io

from assets.symptoms_list import symptoms_list

try:
    from utils.symptom_predictor import get_symptom_prediction
    from utils.image_predictor import get_image_prediction
except ImportError as e:
    st.error(f"Import error: {e}")

st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="centered"
)

# session state for model loading
if 'symptom_model_loaded' not in st.session_state:
    st.session_state.symptom_model_loaded = False
if 'image_model_loaded' not in st.session_state:
    st.session_state.image_model_loaded = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'predicted_disease' not in st.session_state:
    st.session_state.predicted_disease = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "symptom"

# title
st.markdown("<h1 style='text-align: center;'>Disease Predictor</h1>", unsafe_allow_html=True)

# Tabs 
col1, col2 = st.columns([1, 1])
with col1:
    symptom_tab = st.button("Symptom Based", 
                         type="primary" if st.session_state.active_tab == "symptom" else "secondary",
                         use_container_width=True,
                         key="symptom_tab_button")
with col2:
    image_tab = st.button("Image Based", 
                       type="primary" if st.session_state.active_tab == "image" else "secondary",
                       use_container_width=True,
                       key="image_tab_button")

if symptom_tab:
    st.session_state.active_tab = "symptom"
elif image_tab:
    st.session_state.active_tab = "image"

# main content
main_container = st.container()

with main_container:
    if st.session_state.active_tab == "symptom":
        
        if not st.session_state.symptom_model_loaded:
            with st.spinner("Loading symptom prediction model..."):
                
                time.sleep(1)  
                st.session_state.symptom_model_loaded = True
        
        
        st.write("")  
        
    
        symptoms_container = st.container()
        col1, col2, col3 = symptoms_container.columns(3)
        
        
        selected_symptoms = []
        
        
        for i, symptom in enumerate(symptoms_list):
            if i % 3 == 0:
                if col1.checkbox(symptom, key=f"symptom_{i}"):
                    selected_symptoms.append(symptom)
            elif i % 3 == 1:
                if col2.checkbox(symptom, key=f"symptom_{i}"):
                    selected_symptoms.append(symptom)
            else:
                if col3.checkbox(symptom, key=f"symptom_{i}"):
                    selected_symptoms.append(symptom)
        
        # Predict button
        predict_button = st.button("Predict", type="primary", use_container_width=True, key="symptom_predict_button")
        
        if predict_button:
            if len(selected_symptoms) == 0:
                st.error("Please select at least one symptom.")
            else:
                with st.spinner("Predicting disease based on symptoms..."):
                    try:
                        
                        predicted_disease, confidence = get_symptom_prediction(selected_symptoms)
                        
                        
                        st.session_state.predicted_disease = predicted_disease
                        st.session_state.confidence = confidence
                        
                        
                        st.success(f"Prediction: {predicted_disease} (Confidence: {confidence:.2f}%)")
                        
                        
                        st.markdown(f"### About {predicted_disease}")
                        st.write("Consult with a healthcare professional for a proper diagnosis.")
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
    
    else:  # Image-based tab
        if not st.session_state.image_model_loaded:
            with st.spinner("Loading image prediction model..."):
                
                time.sleep(1)  
                st.session_state.image_model_loaded = True
        
        # Image upload 
        uploaded_file = st.file_uploader("Choose an image of the skin condition", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            
            predict_button = st.button("Predict", type="primary", use_container_width=True, key="image_predict_button")
            
            if predict_button:
                with st.spinner("Analyzing image..."):
                    try:
                        # PIL Image to bytes
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        
                        predicted_disease, confidence = get_image_prediction(img_byte_arr)
                        
                        
                        st.session_state.predicted_disease = predicted_disease
                        st.session_state.confidence = confidence
                        
                        
                        st.success(f"Prediction: {predicted_disease} (Confidence: {confidence:.2f}%)")
                        
                        
                        st.markdown(f"### About {predicted_disease}")
                        st.write("Consult with a dermatologist for a proper diagnosis.")
                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")
        else:
            
            st.button("Predict", type="primary", use_container_width=True, disabled=True, key="disabled_predict_button")

# Footer section
st.markdown("---")
st.caption("Disclaimer: This application is for educational purposes only and should not be used for self-diagnosis. Always consult with a qualified healthcare professional for medical advice.")
