import io

import numpy as np
import pytest
from PIL import Image

from classify import (
    CLASS_NAMES,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_FILENAME,
    MODEL_INPUT_SIZE,
    ImageLoadError,
    class_probabilities,
    load_image,
    model_path,
    preprocess,
    preview_image,
    summarize_prediction,
    top_indices,
)


def _png_bytes(image: Image.Image, exif: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    save_kwargs = {"format": "PNG"}
    if exif is not None:
        save_kwargs["exif"] = exif
    image.save(buffer, **save_kwargs)
    return buffer.getvalue()


def test_class_names_match_cifar10():
    assert len(CLASS_NAMES) == 10
    assert CLASS_NAMES[0] == "Airplane"
    assert CLASS_NAMES[-1] == "Truck"
    assert len(set(CLASS_NAMES)) == 10


def test_model_path_uses_committed_filename():
    assert model_path().name == MODEL_FILENAME
    assert model_path("/tmp").as_posix() == f"/tmp/{MODEL_FILENAME}"


def test_load_image_applies_exif_orientation():
    image = Image.new("RGB", (3, 1))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 255, 0))
    image.putpixel((2, 0), (0, 0, 255))
    exif = image.getexif()
    exif[0x0112] = 6  # rotasi 90 derajat searah jarum jam
    loaded = load_image(_png_bytes(image, exif.tobytes()))
    assert loaded.size == (1, 3)
    assert loaded.getpixel((0, 0)) == (255, 0, 0)
    assert loaded.getpixel((0, 2)) == (0, 0, 255)


def test_load_image_rewinds_file_pointer():
    image = Image.new("RGB", (4, 6), (12, 34, 56))
    buffer = io.BytesIO(_png_bytes(image))
    buffer.seek(3)
    loaded = load_image(buffer)
    assert loaded.size == (4, 6)
    assert loaded.getpixel((0, 0)) == (12, 34, 56)


def test_load_image_converts_grayscale_and_alpha():
    gray = load_image(_png_bytes(Image.new("L", (5, 5), 128)))
    assert gray.mode == "RGB"
    assert gray.getpixel((0, 0)) == (128, 128, 128)

    rgba = load_image(_png_bytes(Image.new("RGBA", (5, 5), (10, 20, 30, 128))))
    assert rgba.mode == "RGB"
    assert rgba.getpixel((0, 0)) == (10, 20, 30)


def test_load_image_rejects_unreadable_bytes():
    with pytest.raises(ImageLoadError):
        load_image(b"ini bukan gambar")
    with pytest.raises(ImageLoadError):
        load_image(b"")


def test_preview_limits_the_long_side():
    preview = preview_image(Image.new("RGB", (1000, 400), (1, 2, 3)))
    assert max(preview.size) == 512
    assert preview.size[0] > preview.size[1]


def test_square_image_matches_previous_resize_path():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")

    legacy = image.copy()
    legacy.thumbnail((512, 512))
    legacy = legacy.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    expected = np.expand_dims(np.asarray(legacy, dtype=np.float32) / 255.0, 0)

    np.testing.assert_array_equal(preprocess(image), expected)


def test_wide_image_uses_center_instead_of_stretching_edges():
    image = Image.new("RGB", (64, 32), (255, 0, 0))
    for x in range(16, 48):
        for y in range(32):
            image.putpixel((x, y), (0, 255, 0))

    batch = preprocess(image)
    assert batch.shape == (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)
    assert batch.dtype == np.float32
    assert float(batch.min()) >= 0.0
    assert float(batch.max()) <= 1.0
    assert np.all(batch[0, :, :, 1] > 0.95)
    assert np.all(batch[0, :, :, 0] < 0.05)


def test_top_indices_are_descending():
    probabilities = np.array([0.1, 0.7, 0.2, 0, 0, 0, 0, 0, 0, 0])
    assert top_indices(probabilities, 3) == [1, 2, 0]


def test_summarize_prediction_reports_low_confidence():
    output = np.full((1, 10), 0.1)
    summary = summarize_prediction(output)
    assert summary.label in CLASS_NAMES
    assert summary.confidence == pytest.approx(0.1)
    assert summary.low_confidence
    assert {item.label for item in summary.ranking} == set(CLASS_NAMES)


def test_summarize_prediction_accepts_a_confident_vector():
    output = np.zeros(10)
    output[3] = 1
    summary = summarize_prediction(output)
    assert summary.label == "Cat"
    assert summary.confidence == pytest.approx(1)
    assert not summary.low_confidence
    assert summary.confidence >= LOW_CONFIDENCE_THRESHOLD


def test_preprocess_accepts_grayscale_directly():
    batch = preprocess(Image.new("L", (8, 8), 255))
    assert batch.shape == (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)
    assert np.allclose(batch, 1.0)


def test_load_image_reads_webp():
    image = Image.new("RGB", (8, 8), (9, 8, 7))
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", lossless=True)
    loaded = load_image(buffer.getvalue())
    assert loaded.getpixel((0, 0)) == (9, 8, 7)


def test_class_probabilities_reject_bad_shapes_and_values():
    with pytest.raises(ValueError):
        class_probabilities(np.zeros(9))
    with pytest.raises(ValueError):
        class_probabilities(np.zeros((2, 10)))
    with pytest.raises(ValueError):
        class_probabilities(np.full(10, np.nan))
