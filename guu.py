import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2



model = load_model(r"C:\Users\udhay\emotion_detectionnn.keras")




# Class labels (adjust based on your dataset)
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Streamlit UI
st.title("Facial Expression Recognition")
st.write("Upload an image to classify the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = load_img(uploaded_file, color_mode="grayscale", target_size=(48, 48))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display result
    st.subheader(f"Predicted Emotion: {predicted_class}")
