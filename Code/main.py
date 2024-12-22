import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))

# Path untuk kedua model
mobile_net_path = os.path.join(working_dir, 'UAP_MobileNet.h5')
res_net_path = os.path.join(working_dir, 'UAP_ResNet.h5')

# Load kedua model
try:
    mobile_net_model = tf.keras.models.load_model(mobile_net_path)
    res_net_model = tf.keras.models.load_model(res_net_path)
except AttributeError as e:
    st.error("Failed to load the models. Please check if the model files are corrupted or incompatible.")
    raise e

# Load class indices
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))


# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
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
st.text(
    "Model ini mengklasifikasikan beberapa jenis penyakit pada buah tomat. Tomat (Solanum lycopersicum) "
    "merupakan salah satu tanaman hortikultura yang memiliki peran penting dalam sektor pertanian dan ekonomi. "
    "Produksi tomat yang optimal sangat dipengaruhi oleh kesehatan tanaman, di mana keberadaan penyakit daun pada "
    "tanaman tomat dapat mengurangi hasil panen secara signifikan. Beberapa penyakit umum yang menyerang daun tomat, "
    "seperti Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Septoria Spot, Bacterial Spot, dan Yellow Leaf Curl Virus."
)
st.subheader("Pilih Algoritma")

# Dropdown untuk memilih model
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
            # Pilih model berdasarkan input user
            model = mobile_net_model if model_choice == "MobileNet" else res_net_model
            # Prediksi menggunakan model yang dipilih
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: {str(prediction)}")
