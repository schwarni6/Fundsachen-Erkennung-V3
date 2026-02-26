import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Seiten-Config
st.set_page_config(page_title="Fundsachen KI", page_icon="🔍")

st.title("🔍 Fundsachen-Erkennung")
st.write("Lade ein Bild hoch und die KI erkennt den Gegenstand.")

# Zeigt alle Dateien im Ordner (hilft beim Debuggen)
st.write("📂 Dateien im Projektordner:", os.listdir())

# Modell laden (wird nur einmal geladen)
@st.cache_resource
def load_keras_model():
    model_path = "keras_model.h5"

    if not os.path.exists(model_path):
        st.error(f"❌ Modell-Datei nicht gefunden: {model_path}")
        st.stop()

    return load_model(model_path, compile=False)

model = load_keras_model()

# Labels laden
if not os.path.exists("labels.txt"):
    st.error("❌ labels.txt nicht gefunden!")
    st.stop()

class_names = open("labels.txt", "r").readlines()

# Bild-Upload
uploaded_file = st.file_uploader("📸 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten (wie Teachable Machine es erwartet)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalisieren
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Array vorbereiten
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    st.subheader("📌 Ergebnis:")
    st.success(f"**{class_name}**")

    st.write(f"🔎 Sicherheit: {confidence_score * 100:.2f}%")

    # Zeigt alle Wahrscheinlichkeiten
    st.subheader("📊 Alle Wahrscheinlichkeiten:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i].strip()} → {prob * 100:.2f}%")
