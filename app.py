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


st.set_page_config(page_title="Klasifikasi Gambar CIFAR-10")

st.title("Klasifikasi Gambar CIFAR-10")
st.write(
    "Unggah gambar untuk diprediksi oleh model CNN. "
    "Model mengenali 10 kelas objek."
)

model = get_model()

uploaded_file = st.file_uploader(
    "Pilih gambar", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Gambar diunggah", width=200)

    with col2:
        prediction = model.predict(preprocess(image))[0]
        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[predicted_index])

        st.subheader(class_names[predicted_index])
        st.write(f"Keyakinan: {confidence:.2%}")

        st.write("3 kelas teratas:")
        top_3 = np.argsort(prediction)[::-1][:3]
        for i in top_3:
            st.write(f"{class_names[i]} - {prediction[i]:.2%}")
            st.progress(float(prediction[i]))

st.caption(
    "Kelas yang dikenali: Airplane, Automobile, Bird, Cat, Deer, "
    "Dog, Frog, Horse, Ship, Truck"
)
