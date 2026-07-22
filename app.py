from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]


@st.cache_resource
def get_model():
    return load_model("model_cnn_cifar10_v2.h5")


def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((32, 32))
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)


st.set_page_config(
    page_title="Klasifikasi Gambar CIFAR-10",
    layout="wide"
)

def load_css(path):
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css(Path(__file__).parent / "style.css")

st.title("Klasifikasi Gambar CIFAR-10")
st.write("Model CNN memprediksi objek dari gambar yang diupload.")

model = get_model()

uploaded_file = st.file_uploader(
    "Upload gambar", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(image, caption="Gambar Diupload", use_container_width=True)

    with col2:
        prediction = model.predict(preprocess(image))[0]
        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[predicted_index])

        st.markdown(
            f"""
            <div class="hasil-utama">
                <div class="hasil-label">Terprediksi</div>
                <div class="hasil-kelas">{class_names[predicted_index]}</div>
                <div class="hasil-yakin">Akurasi: {confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="judul-top3">3 kelas teratas</div>',
            unsafe_allow_html=True
        )

        top_3 = np.argsort(prediction)[::-1][:3]
        for i in top_3:
            persen = float(prediction[i]) * 100
            st.markdown(
                f"""
                <div class="bar-label">
                    <span>{class_names[i]}</span>
                    <span>{persen:.2f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {persen:.2f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()
st.markdown(
    """
    <div class="footer-kelas">Kelas yang dikenali: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck</div>
    <div class="footer-identitas">2026 Group 6 - Artificial Intelligence (JTCA)</div>
    """,
    unsafe_allow_html=True
)