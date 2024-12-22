import os
import json
import zipfile
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))

# Path ke file ZIP model
mobile_net_zip = os.path.join(working_dir, 'UAP_MobileNet.zip')
res_net_zip = os.path.join(working_dir, 'UAP_ResNet.zip')

# Fungsi untuk mengekstrak file ZIP dan memuat model
def load_model_from_zip(zip_path, model_name):
    extract_path = os.path.join(working_dir, model_name)
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    model_path = os.path.join(extract_path, model_name + '.h5')
    return tf.keras.models.load_model(model_path)

# Load kedua model dari file ZIP
try:
    mobile_net_model = load_model_from_zip(mobile_net_zip, 'UAP_MobileNet')
    res_net_model = load_model_from_zip(res_net_zip, 'UAP_ResNet')
except Exception as e:
    st.error(f"Failed to load the models: {str(e)}")
    raise e

# Load class indices
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title("Klasifikasi Penyakit pada Tomat Menggunakan Citra Daun")
st.subheader("Pilih Algoritma")

model_choice = st.selectbox(
    "Pilih Model yang Akan Digunakan:",
    ("MobileNet", "ResNet")
)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button("Classify"):
            model = mobile_net_model if model_choice == "MobileNet" else res_net_model
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: {str(prediction)}")
