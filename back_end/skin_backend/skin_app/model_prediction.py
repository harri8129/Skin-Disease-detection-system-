import os 
import numpy as np
from django.conf import settings 
import tensorflow as tf 
from tensorflow.keras.models import load_model # type: ignore
import cv2
from PIL import Image 
from google import genai


STATIC_DIR = settings.STATIC_DIR


model_path = os.path.join(STATIC_DIR,'D:/PROJECTS/computer vision/skin dieases detection/back_end/skin_backend/static/skin_model.h5')
model = tf.keras.models.load_model(model_path)



def pipeline_model(path):
        
        input_shape = model.input_shape[1:4]  # retriving the shape of the model  (256, 256, 3)

        img = Image.open(path)
        img = np.array(img) #converting to numpy bcoz tf/cv works on array

        # Resize the image to the model's expected input size
        img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize to (256, 256)
        
        # Ensure the image has 3 channels (convert grayscale to RGB if necessary)
        if img.ndim == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] != 3:  # Handle cases where the image has alpha channels
            img = img[:, :, :3]

        # Normalize the image
        img = img / 255.0  # Normalize pixel values
        img = img.reshape(1, input_shape[0], input_shape[1], input_shape[2])

        # Make predictions
        prediction = model.predict(img)

        prediction_label = np.argmax(prediction, axis=1)[0]
        labels = {0: 'Actinic keratosis', 1: 'Basal cell carcinoma ', 2: 'Benign keratosis', 3: 'Dermatofibroma',4:'Melanocytic nevus',5:'Melanoma',6:'Squamous cell carcinoma',7:'Vascular lesion'}
        result = labels.get(prediction_label, 'Unknown')


        return result


#integrating gemini 
# Initialize the Gemini client

client = genai.Client(api_key='AIzaSyCU8YuWmW2se9oCzxF4OI55wcbuHd3cuQ4')

# Function to extract diseases details using Gemini
def extract_details(result):
    prompt = f"Explain the skin disease {result}. Include its symptoms, causes, and treatments."

    # Send the request to the Gemini model
    response = client.models.generate_content(
        model='gemini-1.5-pro-002',
        contents=prompt,
    )
    return response.text


#api key     AIzaSyCU8YuWmW2se9oCzxF4OI55wcbuHd3cuQ4