import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
from keras.models import load_model
m

@st.cache(allow_output_mutation=True)
def load_model():
    model = load_model('finalproj.h5')
    return model

if __name__ == "__main__":
    model = load_model()
    if model is not None:
        print("Model loaded successfully!")

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert('RGB')
    img = np.asarray(image)
    img = img / 255.0 
    img_reshape = np.reshape(img, (1, 256, 256, 3))
    prediction = model.predict(img_reshape)
    return prediction
    
model = load_model()

if model is None:
    st.text("Error loading the model. Please try again later.")
else:
    st.write("""
    Detecting AI Generated Art vs Real Art 
    """)

    file = st.file_uploader("Choose any image from computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['AI Generated', 'Not AI Generated']
        string = "This art is most likely: " + class_names[np.argmax(prediction)]
        st.write(string)
