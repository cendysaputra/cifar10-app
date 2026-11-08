"""Foto sungguhan untuk mode Umum.

Lewati di CI. Jalankan lokal dengan model yang sudah di-cache atau jaringan ke Hugging Face:

    GENERAL_LIVE=1 pytest tests/test_general_live.py

Sumber foto ada di tests/fixtures/CREDITS.txt.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import general

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERAL_LIVE") != "1",
    reason="Set GENERAL_LIVE=1 untuk mengunduh CLIP dan menguji foto sungguhan",
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def backend():
    return general.load_backend()


def _predict(backend, name: str):
    with Image.open(FIXTURES / name) as image:
        return backend.predict(image.convert("RGB"))


def test_person_photo_is_a_person(backend):
    prediction = _predict(backend, "person.jpg")
    assert prediction.label in {"Pria", "Orang", "Wanita"}
    assert not prediction.low_confidence
    assert len(prediction.ranking) >= 5
    assert prediction.ranking[0].probability > prediction.ranking[1].probability


def test_house_photo_is_a_house(backend):
    prediction = _predict(backend, "house.jpg")
    assert prediction.label == "Rumah"
    assert not prediction.low_confidence
    assert prediction.ranking[0].probability > prediction.ranking[1].probability


def test_rock_photo_is_a_rock(backend):
    prediction = _predict(backend, "rock.jpg")
    assert prediction.label == "Batu"
    assert not prediction.low_confidence
    assert prediction.ranking[0].probability > prediction.ranking[1].probability


def test_noise_is_not_sure(backend):
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
    prediction = backend.predict(image)
    assert prediction.low_confidence
    assert len(prediction.ranking) >= 5
