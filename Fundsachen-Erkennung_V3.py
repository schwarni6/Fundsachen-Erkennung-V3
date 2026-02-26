import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(page_title="Fundsachen KI", page_icon="🔍")

st.title("🔍 Fundsachen-Erkennung")
st.write("Lade ein Bild hoch und die KI erkennt den Gegenstand.")

@st.cache_resource
def load_keras_model():
    return load_model("keras_model.h5", compile=False)

model = load_keras_model()

class_names = open("labels.txt", "r").readlines()

uploaded_file = st.file_uploader("📸 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    st.subheader("📌 Ergebnis:")
    st.success(f"{class_name}")
    st.write(f"🔎 Sicherheit: {confidence_score * 100:.2f}%")
