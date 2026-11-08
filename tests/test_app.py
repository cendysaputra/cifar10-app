import io

import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from classify import CLASS_NAMES


def _png_bytes() -> bytes:
    image = Image.new("RGB", (64, 32), (30, 90, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _app() -> AppTest:
    app = AppTest.from_file("app.py", default_timeout=60)
    app.run()
    return app


def test_home_explains_the_input_and_lists_classes():
    app = _app()
    assert not app.exception
    assert app.title[0].value == "Klasifikasi Gambar"
    control = app.segmented_control[0]
    assert control.options == ["CIFAR-10", "Umum"]
    assert control.value == "CIFAR-10"
    assert control.proto.label == "Mode klasifikasi"
    assert "32×32" in app.caption[0].value
    allowed = {item.lstrip(".") for item in app.file_uploader[0].allowed_type}
    assert {"jpg", "jpeg", "png", "webp"} <= allowed
    footer = " ".join(block.value for block in app.markdown)
    for name in CLASS_NAMES:
        assert name in footer


def test_valid_upload_shows_a_prediction():
    app = _app()
    app.file_uploader[0].upload("objek.png", _png_bytes(), "image/png")
    app.run()

    assert not app.exception
    assert not app.error
    html = " ".join(block.value for block in app.markdown)
    assert 'role="status"' in html
    assert "Hasil prediksi" in html
    assert any(name in html for name in CLASS_NAMES)
    assert app.expander[0].label == "Semua kelas"
    bar_html = " ".join(
        block.value for block in app.markdown if 'role="progressbar"' in block.value
    )
    assert bar_html.count("bar-track") >= 3
    assert "\n\n" not in bar_html
    for line in bar_html.splitlines():
        if line.strip().startswith("<"):
            assert not line.startswith(" ")


def test_invalid_upload_is_rejected_without_a_traceback():
    app = _app()
    app.file_uploader[0].upload("rusak.png", b"bukan gambar", "image/png")
    app.run()

    assert not app.exception
    assert app.error
    assert "tidak dapat dibaca" in app.error[0].value


def _umum_app(monkeypatch, mock_kind: str) -> AppTest:
    monkeypatch.setenv("GENERAL_MODEL_MOCK", mock_kind)
    st.cache_resource.clear()
    app = AppTest.from_file("app.py", default_timeout=60)
    app.session_state["classifier_mode"] = "Umum"
    app.run()
    if not app.exception and app.segmented_control and app.segmented_control[0].value != "Umum":
        app.segmented_control[0].set_value("Umum")
        app.run()
    return app


def test_umum_mode_shows_top_five(monkeypatch):
    app = _umum_app(monkeypatch, "sure")
    try:
        assert not app.exception
        assert app.segmented_control[0].value == "Umum"
        captions = " ".join(block.value.lower() for block in app.caption)
        assert "bukan model cifar-10" in captions
        app.file_uploader[0].upload("rumah.png", _png_bytes(), "image/png")
        app.run()

        assert not app.exception
        assert not app.error
        html = " ".join(block.value for block in app.markdown)
        status = " ".join(block.value for block in app.markdown if 'role="status"' in block.value)
        assert 'role="status"' in html
        assert "Rumah" in status
        assert "house" in status
        assert "hasil-ragu" not in status
        assert "5 tebakan teratas" in html
        bar_html = " ".join(
            block.value for block in app.markdown if 'role="progressbar"' in block.value
        )
        assert bar_html.count("bar-track") >= 5
        assert app.expander[0].label == "Label yang dikenali"
    finally:
        st.cache_resource.clear()


def test_umum_low_confidence_says_not_sure(monkeypatch):
    app = _umum_app(monkeypatch, "unsure")
    try:
        app.file_uploader[0].upload("abstrak.png", _png_bytes(), "image/png")
        app.run()

        assert not app.exception
        status = " ".join(block.value for block in app.markdown if 'role="status"' in block.value)
        assert "Tidak yakin" in status
        assert "not sure" in status
        assert "hasil-ragu" in status
        assert "Komputer" in status
        bar_html = " ".join(
            block.value for block in app.markdown if 'role="progressbar"' in block.value
        )
        assert bar_html.count("bar-track") >= 5
    finally:
        st.cache_resource.clear()


def test_umum_invalid_upload_is_rejected(monkeypatch):
    app = _umum_app(monkeypatch, "sure")
    try:
        app.file_uploader[0].upload("rusak.png", b"bukan gambar", "image/png")
        app.run()

        assert not app.exception
        assert app.error
        assert "tidak dapat dibaca" in app.error[0].value
    finally:
        st.cache_resource.clear()
