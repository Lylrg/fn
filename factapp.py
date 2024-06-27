import streamlit as st
import pytesseract
from PIL import Image
import io
import numpy as np
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import re
import zipfile
import os
# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

# Path to your compressed model file
compressed_file = 'rf_2.joblib.zip'
# Extracted file name (remove .zip extension)
extracted_file = 'rf_2.joblib'

# Example: Relative path to a directory named 'models' in the current working directory
extract_path = './models/'

# Ensure the extraction path exists
os.makedirs(extract_path, exist_ok=True)

# Extract the compressed file
with zipfile.ZipFile(compressed_file, 'r') as zip_ref:
    zip_ref.extract(extracted_file, extract_path)
# Load the model
model = joblib.load(model_path)



# Load rf model and vectorizer
#model = joblib.load('rf_2.joblib')
cv = joblib.load('cv_2.joblib')

# Set up Tesseract executable path if needed (adjust the path accordingly)
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords (optional)
    stop_words = set(stopwords.words('english'))  # Change to 'spanish' for Spanish text
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to predict and return label
def predict_label(text):
    processed_text = preprocess_text(text)
    X_transformed = cv.transform([processed_text])
    prediction = model.predict(X_transformed)[0]
    return "Real" if prediction == 1 else "Fake"

# Streamlit app
def main():
    st.title("FactSniffer: Fake News Detector")
    my_expander = st.expander(label='Get Started 👋 🚀')
    with my_expander:
        'Using FactSniffer is simple! Just drop in the news article or tweet you’re suspicious of, and let our model do the heavy lifting. 📰💻 In no time, you’ll know if it’s legit or a fraud. 🚨 Join the anti-misinformation squad and become the Sherlock Holmes of the digital age! 🔍'

    # Sidebar with radio button for input options
    input_option = st.sidebar.radio(
            "Choose the way to contrast your information and let FactSniffer do the rest! 📰💻🔍",
            ['Upload an image', 'Enter text manually']
        )

    st.sidebar.subheader("Extras")
    if st.sidebar.button("Show Tips on Spotting Fake News"):
        st.sidebar.write("""
            1. Check the source.
            2. Look for unusual formatting.
            3. Inspect the dates.
            4. Verify with other sources.
            5. Be skeptical of sensational headlines.
        """)
    
    
    # Main content area with two columns
    col1, col2 = st.columns([1, 1])

    # Column 1: Upload an image
    with col1:
        if input_option == 'Upload an image':
            #st.header("Upload an Image")
            uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption='Uploaded Image', use_column_width=True)
        elif input_option == 'Enter text manually':
            #st.header("Enter Text Manually")
            text_input = st.text_area('Enter text to classify:')
            if st.button('Classify'):
                if text_input:
                    processed_text = preprocess_text(text_input)
                    prediction_label = predict_label(processed_text)
                    if prediction_label == "Real":
                        st.success(" ✅ Trustworthy ")
                    else:
                        st.error("❌ False")
    
    # Column 2: Display prediction result
    with col2:
        if input_option == 'Upload an image' and uploaded_image:
            st.header("Prediction")
            extracted_text = pytesseract.image_to_string(image)
            processed_text = preprocess_text(extracted_text)
            prediction_label = predict_label(processed_text)
            if prediction_label == "Real":
                st.success("✅ Trustworthy ")
            else:
                st.error("❌ False")

        

if __name__ == '__main__':
    main()

    