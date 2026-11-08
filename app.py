"""Aplikasi Streamlit untuk klasifikasi gambar CIFAR-10."""

from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from classify import (
    CLASS_NAMES,
    ImageLoadError,
    load_image,
    model_path,
    preprocess,
    preview_image,
    summarize_prediction,
)

st.set_page_config(
    page_title="Klasifikasi Gambar CIFAR-10",
    layout="wide",
)


def load_css(path: Path) -> None:
    if not path.is_file():
        return
    css = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def predict(model, image: Image.Image):
    raw = np.asarray(model(preprocess(image), training=False))
    return summarize_prediction(raw)


@st.cache_resource
def get_model():
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    from tensorflow.keras.models import load_model

    return load_model(str(path), compile=False)


load_css(Path(__file__).resolve().parent / "style.css")

st.title("Klasifikasi Gambar CIFAR-10")
st.write("Model CNN memprediksi objek dari gambar yang diupload.")

model = None
try:
    model = get_model()
except FileNotFoundError:
    st.error(
        "Model tidak dapat dimuat. Pastikan berkas "
        f"`{model_path().name}` ada di folder aplikasi, lalu muat ulang halaman."
    )
except Exception:
    st.error("Model tidak dapat dimuat. Muat ulang halaman, lalu coba lagi.")

uploaded_file = None
if model is not None:
    uploaded_file = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png", "webp"],
    )

if uploaded_file is not None:
    try:
        image = load_image(uploaded_file.getvalue())
        preview = preview_image(image)
        prediction = predict(model, image)
    except ImageLoadError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Prediksi gagal untuk gambar ini. Coba file JPG, PNG, atau WEBP lain.")
    else:
        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.image(preview, caption="Gambar Diupload", width="stretch")

        with col2:
            st.markdown(
                f"""
                <div class="hasil-utama">
                    <div class="hasil-label">Terprediksi</div>
                    <div class="hasil-kelas">{escape(prediction.label)}</div>
                    <div class="hasil-yakin">Probabilitas: {prediction.confidence:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="judul-top3">3 kelas teratas</div>',
                unsafe_allow_html=True,
            )

            for item in prediction.ranking[:3]:
                persen = max(0.0, min(100.0, item.probability * 100))
                st.markdown(
                    f"""
                    <div class="bar-label">
                        <span>{escape(item.label)}</span>
                        <span>{persen:.2f}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {persen:.2f}%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.divider()
st.markdown(
    f"""
    <div class="footer-kelas">Kelas yang dikenali: {escape(", ".join(CLASS_NAMES))}</div>
    <div class="footer-identitas">2026 Group 6 - Artificial Intelligence (JTCA)</div>
    """,
    unsafe_allow_html=True,
)
