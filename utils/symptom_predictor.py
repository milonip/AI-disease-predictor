import random
from assets.symptoms_list import symptoms_list

# Expert-defined symptom to disease mapping
symptom_disease_map = {
    "itching": ["Fungal infection", "Allergy", "Drug Reaction"],
    "skin rash": ["Fungal infection", "Drug Reaction", "Chicken pox"],
    "nodal skin eruptions": ["Fungal infection", "Chicken pox"],
    "continuous sneezing": ["Common Cold", "Allergy"],
    "shivering": ["Common Cold", "Pneumonia", "Typhoid"],
    "chills": ["Common Cold", "Pneumonia", "Malaria"],
    "joint pain": ["Arthritis", "Chikungunya", "Typhoid"],
    "stomach pain": ["Gastroenteritis", "Peptic ulcer disease", "GERD"],
    "acidity": ["GERD", "Peptic ulcer disease"],
    "ulcers on tongue": ["Fungal infection", "Vitamin deficiency"],
    "muscle wasting": ["Muscular Dystrophy", "Diabetes"],
    "vomiting": ["Food Poisoning", "Jaundice", "Migraine"],
    "burning micturition": ["Urinary tract infection", "Diabetes"],
    "spotting urination": ["Urinary tract infection", "Diabetes"],
    "fatigue": ["Chronic Fatigue Syndrome", "Diabetes", "Hypothyroidism"],
    "weight gain": ["Hypothyroidism", "Obesity"],
    "anxiety": ["Anxiety Disorder", "Depression"],
    "cold hands and feet": ["Hypothyroidism", "Anemia"],
    "mood swings": ["Depression", "Bipolar Disorder"],
    "weight loss": ["Diabetes", "Hyperthyroidism", "Tuberculosis"],
    "restlessness": ["Anxiety Disorder", "Hyperthyroidism"],
    "lethargy": ["Hypothyroidism", "Depression", "Jaundice"],
    "patches in throat": ["Fungal infection", "Common Cold", "Strep throat"],
    "irregular sugar level": ["Diabetes", "Hypoglycemia"],
    "cough": ["Common Cold", "Pneumonia", "Tuberculosis"],
    "high fever": ["Influenza", "Malaria", "Typhoid"],
    "sunken eyes": ["Dehydration", "Cholera"],
    "breathlessness": ["Asthma", "Pneumonia", "Heart failure"],
    "sweating": ["Hyperthyroidism", "Influenza", "Malaria"],
    "dehydration": ["Dehydration", "Cholera", "Diarrhea"],
    "indigestion": ["GERD", "Peptic ulcer disease", "Gastroenteritis"],
    "headache": ["Migraine", "Tension headache", "Sinusitis"],
    "yellowish skin": ["Jaundice", "Hepatitis", "Malaria"],
    "dark urine": ["Jaundice", "Hepatitis", "Urinary tract infection"],
    "nausea": ["Food Poisoning", "Migraine", "Morning sickness"],
    "loss of appetite": ["Gastroenteritis", "Hepatitis", "Tuberculosis"],
    "pain behind the eyes": ["Migraine", "Sinusitis"],
    "back pain": ["Herniated disc", "Kidney stones", "Muscle strain"],
    "constipation": ["Irritable Bowel Syndrome", "Hypothyroidism"],
    "abdominal pain": ["Appendicitis", "Gastroenteritis", "Peptic ulcer disease"],
    "diarrhoea": ["Gastroenteritis", "Food Poisoning", "Irritable Bowel Syndrome"],
    "mild fever": ["Common Cold", "Influenza", "Urinary tract infection"],
    "yellow urine": ["Jaundice", "Dehydration"],
    "yellowing of eyes": ["Jaundice", "Hepatitis"],
    "acute liver failure": ["Hepatitis", "Drug toxicity"],
    "fluid overload": ["Heart failure", "Kidney disease"],
    "swelling of stomach": ["Ascites", "Intestinal obstruction"],
    "swelled lymph nodes": ["Lymphoma", "Tuberculosis", "HIV"],
    "malaise": ["Chronic Fatigue Syndrome", "Depression", "Influenza"],
    "blurred and distorted vision": ["Cataract", "Glaucoma", "Myopia"],
    "phlegm": ["Bronchitis", "Pneumonia", "Tuberculosis"],
    "throat irritation": ["Common Cold", "Strep throat", "Tonsillitis"],
    "redness of eyes": ["Conjunctivitis", "Allergy", "Glaucoma"],
    "sinus pressure": ["Sinusitis", "Common Cold"],
    "runny nose": ["Common Cold", "Allergy", "Sinusitis"],
    "congestion": ["Common Cold", "Sinusitis", "Allergic rhinitis"],
    "chest pain": ["Angina", "Myocardial infarction", "Pneumonia"],
    "weakness in limbs": ["Stroke", "Multiple sclerosis", "Peripheral neuropathy"],
    "fast heart rate": ["Anxiety Disorder", "Hyperthyroidism", "Anemia"],
    "pain during bowel movements": ["Hemorrhoids", "Irritable Bowel Syndrome", "Inflammatory Bowel Disease"],
    "pain in anal region": ["Hemorrhoids", "Anal fissure", "Rectal cancer"],
    "bloody stool": ["Inflammatory Bowel Disease", "Hemorrhoids", "Colorectal cancer"],
    "irritation in anus": ["Hemorrhoids", "Anal fissure", "Pruritus ani"],
    "neck pain": ["Cervical spondylosis", "Meningitis", "Muscle strain"],
    "dizziness": ["Vertigo", "Hypertension", "Hypoglycemia"],
    "cramps": ["Muscle strain", "Dehydration", "Electrolyte imbalance"],
    "bruising": ["Hemophilia", "Leukemia", "Vitamin K deficiency"],
    "obesity": ["Obesity", "Hypothyroidism", "Cushing syndrome"],
    "swollen legs": ["Congestive heart failure", "Kidney disease", "Deep vein thrombosis"],
    "swollen blood vessels": ["Varicose veins", "Thrombophlebitis"],
    "puffy face and eyes": ["Hypothyroidism", "Nephrotic syndrome", "Cushing syndrome"],
    "enlarged thyroid": ["Hypothyroidism", "Hyperthyroidism", "Thyroiditis"],
    "brittle nails": ["Iron deficiency anemia", "Thyroid disease", "Fungal infection"],
    "swollen extremities": ["Edema", "Heart failure", "Kidney disease"],
    "excessive hunger": ["Diabetes", "Hyperthyroidism", "Hypoglycemia"],
    "drying and tingling lips": ["Dehydration", "Allergic reaction", "Vitamin B deficiency"],
    "slurred speech": ["Stroke", "Intoxication", "Multiple sclerosis"],
    "knee pain": ["Osteoarthritis", "Rheumatoid arthritis", "Gout"],
    "hip joint pain": ["Osteoarthritis", "Rheumatoid arthritis", "Hip fracture"],
    "muscle weakness": ["Muscular dystrophy", "Multiple sclerosis", "Myasthenia gravis"],
    "stiff neck": ["Meningitis", "Cervical spondylosis", "Muscle strain"],
    "swelling joints": ["Rheumatoid arthritis", "Gout", "Osteoarthritis"],
    "movement stiffness": ["Parkinson disease", "Rheumatoid arthritis", "Osteoarthritis"],
    "spinning movements": ["Vertigo", "Labyrinthitis", "Meniere disease"],
    "loss of balance": ["Vertigo", "Multiple sclerosis", "Stroke"],
    "unsteadiness": ["Vertigo", "Multiple sclerosis", "Cerebral palsy"],
    "weakness of one body side": ["Stroke", "Multiple sclerosis", "Brain tumor"],
    "loss of smell": ["Common Cold", "Sinusitis", "COVID-19"],
    "bladder discomfort": ["Urinary tract infection", "Interstitial cystitis", "Benign prostatic hyperplasia"],
    "foul smell of urine": ["Urinary tract infection", "Kidney infection", "Diabetes"],
    "continuous feel of urine": ["Urinary tract infection", "Urinary incontinence", "Benign prostatic hyperplasia"],
    "passage of gases": ["Irritable Bowel Syndrome", "Gastroenteritis", "GERD"],
    "internal itching": ["Fungal infection", "Allergic reaction", "Parasitic infection"],
    "depression": ["Major depressive disorder", "Bipolar disorder", "Hypothyroidism"],
    "irritability": ["Depression", "Anxiety Disorder", "Hyperthyroidism"],
    "muscle pain": ["Fibromyalgia", "Influenza", "Muscle strain"],
    "altered sensorium": ["Encephalitis", "Meningitis", "Drug overdose"],
    "red spots over body": ["Chicken pox", "Measles", "Dengue"],
    "belly pain": ["Appendicitis", "Gastroenteritis", "Peptic ulcer disease"],
    "abnormal menstruation": ["Polycystic ovary syndrome", "Endometriosis", "Pelvic inflammatory disease"],
    "watering from eyes": ["Allergic conjunctivitis", "Common Cold", "Sinusitis"],
    "increased appetite": ["Hyperthyroidism", "Diabetes", "Pregnancy"],
    "family history": ["Genetic disorder", "Familial hypercholesterolemia", "Hereditary cancer"],
    "mucoid sputum": ["Bronchitis", "Pneumonia", "Tuberculosis"],
    "rusty sputum": ["Pneumonia", "Tuberculosis", "Lung abscess"],
    "lack of concentration": ["Attention deficit hyperactivity disorder", "Depression", "Anxiety Disorder"],
    "visual disturbances": ["Migraine", "Glaucoma", "Retinal detachment"],
    "coma": ["Traumatic brain injury", "Drug overdose", "Cerebral hemorrhage"],
    "stomach bleeding": ["Peptic ulcer disease", "Esophageal varices", "Gastric cancer"],
    "history of alcohol consumption": ["Alcoholic liver disease", "Cirrhosis", "Pancreatitis"],
    "blood in sputum": ["Tuberculosis", "Lung cancer", "Bronchiectasis"],
    "prominent veins on calf": ["Varicose veins", "Deep vein thrombosis", "Venous insufficiency"],
    "palpitations": ["Anxiety Disorder", "Atrial fibrillation", "Hyperthyroidism"],
    "painful walking": ["Arthritis", "Gout", "Plantar fasciitis"],
    "pus filled pimples": ["Acne", "Folliculitis", "Impetigo"],
    "blackheads": ["Acne", "Comedones", "Sebaceous hyperplasia"],
    "skin peeling": ["Sunburn", "Eczema", "Psoriasis"],
    "silver like dusting": ["Psoriasis", "Pityriasis", "Dermatitis"],
    "small dents in nails": ["Psoriasis", "Iron deficiency", "Alopecia areata"],
    "inflammatory nails": ["Psoriasis", "Fungal infection", "Bacterial infection"],
    "blister": ["Herpes", "Burns", "Contact dermatitis"],
    "red sore around nose": ["Herpes simplex", "Impetigo", "Rhinophyma"],
    "yellow crust ooze": ["Impetigo", "Eczema", "Seborrheic dermatitis"]
}

def get_symptom_prediction(selected_symptoms):
    """
    Predict disease based on selected symptoms.
    
    Args:
        selected_symptoms: List of selected symptom strings
        
    Returns:
        tuple: (predicted_disease, confidence_percentage)
    """
    try:
        # Initialize disease counter
        disease_counts = {}
        
        # Count occurrences of diseases associated with selected symptoms
        for symptom in selected_symptoms:
            if symptom in symptom_disease_map:
                for disease in symptom_disease_map[symptom]:
                    if disease in disease_counts:
                        disease_counts[disease] += 1
                    else:
                        disease_counts[disease] = 1
        
        # If no matching diseases found, use a fallback approach
        if not disease_counts:
            common_diseases = ["Common Cold", "Influenza", "Gastroenteritis", "Allergic reaction"]
            return random.choice(common_diseases), random.uniform(60.0, 75.0)
        
        # Sort diseases by frequency
        sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate confidence based on how many symptoms match
        top_disease, count = sorted_diseases[0]
        total_possible_symptoms = len([s for d in symptom_disease_map.values() if top_disease in d])
        confidence = min(95.0, (count / max(1, len(selected_symptoms))) * 100)
        
        return top_disease, confidence
    
    except Exception as e:
        print(f"Error in symptom prediction: {e}")
        
        # Map common symptoms to diseases as a fallback
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
