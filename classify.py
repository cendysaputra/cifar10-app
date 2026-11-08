"""Helper klasifikasi CIFAR-10 yang tidak bergantung pada Streamlit.

Model mengharapkan gambar RGB 32x32 dengan piksel 0-1, sama seperti train.py.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_NAMES: tuple[str, ...] = (
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
)

MODEL_FILENAME = "model_cnn_cifar10_v5.h5"
MODEL_INPUT_SIZE = 32
PREVIEW_MAX_SIDE = 512
# Batasi decode agar file kecil tidak berkembang menjadi gambar raksasa.
MAX_IMAGE_PIXELS = 40_000_000
LOW_CONFIDENCE_THRESHOLD = 0.5

_UNREADABLE = "Gambar tidak dapat dibaca. Gunakan file JPG, PNG, atau WEBP yang valid."


class ImageLoadError(ValueError):
    """Berkas unggahan tidak bisa dipakai sebagai gambar."""


@dataclass(frozen=True)
class RankedClass:
    label: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    low_confidence: bool
    ranking: tuple[RankedClass, ...]


def model_path(base_dir: Path | None = None) -> Path:
    root = Path(__file__).resolve().parent if base_dir is None else Path(base_dir)
    return root / MODEL_FILENAME


def load_image(source: bytes | bytearray | io.BufferedIOBase) -> Image.Image:
    """Buka unggahan, terapkan orientasi EXIF, lalu kembalikan gambar RGB."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(source) as opened:
            opened.load()
            transposed = ImageOps.exif_transpose(opened) or opened
            rgb = transposed.convert("RGB")
            image = rgb.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise ImageLoadError(_UNREADABLE) from exc

    if image.width < 1 or image.height < 1:
        raise ImageLoadError(_UNREADABLE)
    return image


def preview_image(image: Image.Image, max_side: int = PREVIEW_MAX_SIDE) -> Image.Image:
    """Salinan untuk ditampilkan, sisi terpanjang paling banyak `max_side` piksel."""
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    return preview


def center_square(image: Image.Image) -> Image.Image:
    """Crop persegi dari tengah agar resize tidak meregangkan foto."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    # copy() melepaskan crop dari piksel asli supaya preview tidak berubah.
    return image.crop((left, top, left + side, top + side)).copy()


def preprocess(image: Image.Image) -> np.ndarray:
    """Batch (1, 32, 32, 3) dengan nilai piksel di rentang 0-1."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    prepared = center_square(image)
    if max(prepared.size) > PREVIEW_MAX_SIDE:
        prepared.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    resized = prepared.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if array.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3):
        raise ImageLoadError("Gambar tidak dapat diubah ke ukuran masukan model.")
    return array[np.newaxis, ...]


def horizontal_flip(batch: np.ndarray) -> np.ndarray:
    """Salinan batch NHWC yang dicerminkan pada sumbu lebar."""
    array = np.asarray(batch)
    if array.ndim != 4:
        raise ValueError("Batch cermin harus berbentuk (N, tinggi, lebar, kanal).")
    return np.ascontiguousarray(np.flip(array, axis=2))


def average_probabilities(views: np.ndarray) -> np.ndarray:
    """Rata-rata beberapa vektor 10 kelas, misalnya gambar asli dan cerminnya."""
    rows = np.asarray(views, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Rata-rata membutuhkan bentuk (tampilan, {len(CLASS_NAMES)}), "
            f"diterima {tuple(np.asarray(views).shape)}."
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("Probabilitas berisi nilai yang tidak valid.")
    return rows.mean(axis=0)


def class_probabilities(output: np.ndarray) -> np.ndarray:
    """Vektor 10 probabilitas dari keluaran model berbentuk (10,) atau (1, 10)."""
    probs = np.asarray(output, dtype=np.float64)
    expected = (len(CLASS_NAMES),)
    if probs.shape == (1, len(CLASS_NAMES)):
        probs = probs[0]
    if probs.shape != expected:
        raise ValueError(
            f"Model mengembalikan bentuk {tuple(np.asarray(output).shape)}, "
            f"diharapkan {expected} atau (1, {len(CLASS_NAMES)})."
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model mengembalikan nilai yang tidak valid.")
    return probs


def top_indices(probabilities: np.ndarray, k: int = 3) -> list[int]:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size == 0 or k <= 0:
        return []
    order = np.argsort(probs)[::-1]
    return [int(index) for index in order[: min(k, probs.size)]]


def summarize_prediction(output: np.ndarray) -> Prediction:
    probs = class_probabilities(output)
    ranking = tuple(
        RankedClass(CLASS_NAMES[index], float(probs[index]))
        for index in top_indices(probs, len(CLASS_NAMES))
    )
    top = ranking[0]
    return Prediction(
        label=top.label,
        confidence=top.probability,
        low_confidence=top.probability < LOW_CONFIDENCE_THRESHOLD,
        ranking=ranking,
    )
