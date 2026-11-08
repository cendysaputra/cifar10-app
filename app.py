"""Aplikasi Streamlit untuk klasifikasi gambar CIFAR-10 dan mode Umum."""

from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from classify import (
    CLASS_NAMES,
    ImageLoadError,
    average_probabilities,
    horizontal_flip,
    load_image,
    model_path,
    preprocess,
    preview_image,
    summarize_prediction,
)

MODE_CIFAR = "CIFAR-10"
MODE_UMUM = "Umum"

st.set_page_config(
    page_title="Klasifikasi Gambar",
    page_icon="🖼️",
    layout="wide",
)


def load_css(path: Path) -> None:
    if not path.is_file():
        return
    css = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def probability_bar_html(label: str, probability: float) -> str:
    """HTML satu batang. Tanpa indentasi, agar Markdown tidak mengubahnya jadi kode."""
    persen = max(0.0, min(100.0, probability * 100))
    safe = escape(label)
    return (
        '<div class="bar-label">'
        f"<span>{safe}</span><span>{persen:.2f}%</span></div>"
        '<div class="bar-track" role="progressbar" aria-valuemin="0" '
        f'aria-valuemax="100" aria-valuenow="{persen:.1f}" aria-label="{safe}">'
        f'<div class="bar-fill" style="width: {persen:.2f}%;"></div></div>'
    )


def predict(model, image: Image.Image):
    batch = preprocess(image)
    views = np.concatenate([batch, horizontal_flip(batch)], axis=0)
    raw = np.asarray(model(views, training=False))
    return summarize_prediction(average_probabilities(raw))


@st.cache_data(show_spinner=False)
def classify_upload(_model, payload: bytes):
    image = load_image(payload)
    return preview_image(image), predict(_model, image)


@st.cache_data(show_spinner=False)
def classify_upload_general(payload: bytes, backend_id: str, _backend):
    del backend_id
    image = load_image(payload)
    return preview_image(image), _backend.predict(image)


@st.cache_resource
def get_model():
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    from tensorflow.keras.models import load_model

    return load_model(str(path), compile=False)


@st.cache_resource(show_spinner=False)
def get_general_backend():
    import general

    return general.load_backend()


def selected_mode() -> str:
    raw = st.segmented_control(
        "Mode klasifikasi",
        options=[MODE_CIFAR, MODE_UMUM],
        default=MODE_CIFAR,
        key="classifier_mode",
        required=True,
        width="stretch",
        help=(
            "CIFAR-10 memakai model 10 kelas yang sudah ada di aplikasi. "
            "Umum mengenali ratusan benda sehari-hari dan mengunduh model sekali."
        ),
    )
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw not in (MODE_CIFAR, MODE_UMUM):
        return MODE_CIFAR
    return str(raw)


def release_other_model(active: str) -> None:
    """Lepas model yang tidak dipakai agar kedua framework tidak menginap bersama."""
    previous = st.session_state.get("loaded_model_kind")
    if previous == active:
        return
    if previous == MODE_CIFAR:
        get_model.clear()
    elif previous == MODE_UMUM:
        get_general_backend.clear()
    if previous is not None:
        import gc

        gc.collect()


def remember_model(active: str) -> None:
    st.session_state["loaded_model_kind"] = active


def render_prediction(
    preview: Image.Image,
    prediction,
    *,
    image_note: str,
    heading: str,
    top_n: int,
    headline: str,
    unsure_html: str,
    english: str | None,
    all_scores: bool,
) -> None:
    bars = [
        f'<div class="judul-peringkat" id="judul-peringkat">{escape(heading)}</div>',
        '<div role="group" aria-labelledby="judul-peringkat">',
    ]
    for item in prediction.ranking[:top_n]:
        bars.append(probability_bar_html(item.label, item.probability))
    bars.append("</div>")
    # Satu baris HTML, tanpa baris kosong: Markdown menutup blok HTML di baris kosong
    # dan menampilkan sisa tag sebagai kode.
    card = [
        '<div class="hasil-utama" role="status" aria-live="polite">',
        '<div class="hasil-label">Hasil prediksi</div>',
        f'<div class="hasil-kelas">{escape(headline)}</div>',
    ]
    if english:
        card.append(f'<div class="hasil-en">{escape(english)}</div>')
    card.append(
        f'<div class="hasil-yakin">Probabilitas: {prediction.confidence:.2%}</div>'
    )
    if unsure_html:
        card.append(unsure_html)
    card.append("</div>")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.image(preview, caption="Gambar yang diunggah", width="stretch")
        st.caption(image_note)
    with col2:
        st.markdown("".join(card), unsafe_allow_html=True)
        st.markdown("".join(bars), unsafe_allow_html=True)
        if all_scores:
            with st.expander("Semua kelas"):
                for item in prediction.ranking:
                    st.write(f"{item.label}: {item.probability:.2%}")


def render_footer(class_line: str) -> None:
    st.divider()
    st.markdown(
        f"""
        <div class="footer-kelas">{class_line}</div>
        """,
        unsafe_allow_html=True,
    )


def show_upload_error(exc: Exception) -> None:
    if isinstance(exc, ImageLoadError):
        st.error(str(exc))
        return
    message = getattr(exc, "args", ("",))
    if exc.__class__.__name__ == "GeneralModelError" and message and message[0]:
        st.error(str(exc))
        return
    st.error("Prediksi gagal untuk gambar ini. Coba file JPG, PNG, atau WEBP lain.")


load_css(Path(__file__).resolve().parent / "style.css")

st.title("Klasifikasi Gambar")
mode = selected_mode()
release_other_model(mode)

uploaded_file = None
if mode == MODE_CIFAR:
    st.write("Model CNN memprediksi objek dari gambar yang diunggah.")
    st.caption(
        "Gambar diubah menjadi 32×32 piksel. Foto yang tidak persegi di-crop dari tengah, "
        "jadi objek sebaiknya berada di tengah bingkai. Prediksi merata-ratakan gambar "
        "dengan cermin horizontalnya."
    )
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
    else:
        remember_model(MODE_CIFAR)
    if model is not None:
        uploaded_file = st.file_uploader(
            "Unggah gambar",
            type=["jpg", "jpeg", "png", "webp"],
            help="File JPG, PNG, atau WEBP, maksimal 5 MB.",
            key="unggahan",
        )
    if uploaded_file is not None and model is not None:
        try:
            with st.spinner("Memprediksi kelas objek..."):
                preview, prediction = classify_upload(model, uploaded_file.getvalue())
        except Exception as exc:
            show_upload_error(exc)
        else:
            unsure = ""
            if prediction.low_confidence:
                unsure = (
                    '<p class="hasil-ragu">Keyakinan model di bawah 50%. '
                    "Prediksi bisa meleset jika objek tidak di tengah atau bukan "
                    "salah satu kelas yang dikenali.</p>"
                )
            render_prediction(
                preview,
                prediction,
                image_note="Model melihat crop tengah 32×32 dan cermin horizontalnya.",
                heading="3 kelas teratas",
                top_n=3,
                headline=prediction.label,
                unsure_html=unsure,
                english=None,
                all_scores=True,
            )
    render_footer(
        "Kelas yang dikenali: " + escape(", ".join(CLASS_NAMES))
    )
else:
    import general
    import general_labels

    st.write(
        "Mode Umum menebak foto ke dalam ratusan label sehari-hari: orang, rumah, batu, "
        "hewan, makanan, kendaraan, benda, dan pemandangan."
    )
    st.caption(
        "Ini bukan model CIFAR-10. Probabilitas tersebar ke banyak label, jadi angka "
        "tertinggi bisa di bawah 50% meski tebakannya tepat. Gambar yang buram, abstrak, "
        "atau tidak mirip label mana pun ditandai tidak yakin."
    )
    backend = None
    try:
        if general.vision_weights_cached():
            spinner = "Memuat model umum..."
        else:
            spinner = "Mengunduh model umum (sekitar 85 MB). Unduhan ini sekali per server."
        with st.spinner(spinner):
            backend = get_general_backend()
    except general.GeneralModelError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Model umum tidak dapat dimuat. Muat ulang halaman, lalu coba lagi.")
    else:
        remember_model(MODE_UMUM)
    if backend is not None:
        uploaded_file = st.file_uploader(
            "Unggah gambar",
            type=["jpg", "jpeg", "png", "webp"],
            help="File JPG, PNG, atau WEBP, maksimal 5 MB.",
            key="unggahan",
        )
    if uploaded_file is not None and backend is not None:
        try:
            with st.spinner("Membandingkan gambar dengan label umum..."):
                preview, prediction = classify_upload_general(
                    uploaded_file.getvalue(),
                    backend.cache_id,
                    backend,
                )
        except Exception as exc:
            show_upload_error(exc)
        else:
            if prediction.low_confidence:
                headline = "Tidak yakin"
                english = None
                unsure = (
                    '<p class="hasil-ragu">Tidak yakin (not sure). '
                    "Gambar ini tidak cukup dekat dengan label yang dikenali, "
                    "atau hampir seri dengan tebakan lain. "
                    f"Tebakan terdekat: {escape(prediction.label)} "
                    f"({prediction.confidence:.2%}).</p>"
                )
            else:
                headline = prediction.label
                english = general.english_name(prediction.label)
                unsure = ""
            render_prediction(
                preview,
                prediction,
                image_note="Model memakai bagian tengah foto berukuran 224×224 piksel.",
                heading="5 tebakan teratas",
                top_n=5,
                headline=headline,
                unsure_html=unsure,
                english=english,
                all_scores=False,
            )
    if backend is not None:
        with st.expander("Label yang dikenali"):
            st.caption(
                "Nama di aplikasi memakai bahasa Indonesia. "
                "Model membandingkan foto dengan kalimat bahasa Inggris untuk tiap label."
            )
            for title, labels in general_labels.labels_by_category():
                names = ", ".join(label.name_id for label in labels)
                st.markdown(f"**{title}**")
                st.write(names)
    render_footer(
        "Mode Umum: "
        f"{len(general_labels.LABELS)} label sehari-hari, termasuk orang, rumah, dan batu. "
        "Bukan daftar tertutup 10 kelas."
    )
