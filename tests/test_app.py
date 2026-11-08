import io

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
    assert app.title[0].value == "Klasifikasi Gambar CIFAR-10"
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
