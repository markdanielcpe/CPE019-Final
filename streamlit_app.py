import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('main.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['png', 'jpg', 'jpeg'])

def preprocess_image(image):
    img = image.resize((256, 256))  # Adjust the size to match your model's input
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_image(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        class_index = 0
    else:
        class_index = 1

    if class_index == 0:
        return 'Ambulance'
    else:
        return 'Vehicle'

def main():
    st.title("Ambulance and Car Image Classification")
    st.text("Upload an image and the model will determine if the image displays an emergency \nvehicle used for transporting patients or a standard automobile.")

    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        result = classify_image(image)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        if result == 'Vehicle':
            st.write(f"The uploaded image is most likely: {result}")
        else:
            st.write(f"The uploaded image is most likely: {result}")
        
if __name__ == '__main__':
    main()
