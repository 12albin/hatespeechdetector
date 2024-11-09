import nltk
import os

# Specify the directory where you want to store NLTK data
nltk_data_dir = 'C:/Users/Albin T S/Desktop/AIML project/hate_speech/nltk_data'  # Replace with your desired path

# Create the directory if it doesn't exist
os.makedirs(nltk_data_dir, exist_ok=True)

# Set the NLTK data path to the specified directory
nltk.data.path.append(nltk_data_dir)

# Download the 'punkt' tokenizer
nltk.download('punkt', download_dir=nltk_data_dir)

# Check that 'punkt' was downloaded successfully
print(f"punkt resource downloaded to: {nltk_data_dir}")
