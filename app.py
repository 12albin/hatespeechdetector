import tensorflow as tf
import numpy as np
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import streamlit as stl
import time
import joblib
import nltk
import os
# Step 1: Specify the directory where 'punkt' is stored

nltk_data_dir = 'nltk_data'  # Your custom path
os.makedirs(nltk_data_dir, exist_ok=True)  # Ensure the directory exists

# Step 2: Set NLTK data path to your custom directory
nltk.data.path.append(nltk_data_dir)


stl.title("Hate Speech Detector")
class_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
model = tf.keras.models.load_model('text_classification_model.h5')
def custom_tokenizer(tokens):
    return tokens 
vectorizer = joblib.load('tfidf_vectorizer.pkl')
 
def clean_text(text):

    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower() 
    return text

def predict_new_input(new_input):
    new_input_cleaned = clean_text(new_input)
    new_input_tokens = word_tokenize(new_input_cleaned)
    new_input_tfidf = vectorizer.transform([new_input_tokens])
    y_pred_new = model.predict(new_input_tfidf)
    y_pred_class = np.argmax(y_pred_new, axis=1)
    predicted_label = class_labels[y_pred_class[0]]
    return predicted_label


tweet = stl.text_input("Enter the sentence:")
submit = stl.button("Predict")
if submit:
        predicted_class = predict_new_input(tweet)
        stl.write(f"Predicted class: {predicted_class}")
   
