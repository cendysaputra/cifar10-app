"""Klasifikasi zero-shot untuk mode Umum.

Foto dibanding dengan embedding teks CLIP ViT-B/32 yang sudah dihitung
untuk daftar label. Bobot vision diunduh saat runtime dan tidak masuk git.
TensorFlow tidak diimpor di sini, jadi mode CIFAR-10 tidak memuat model ini.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from classify import Prediction, RankedClass
from general_labels import LABELS, NULL_PROMPTS, label_by_id

VISION_FILENAME = "clip-vit-b32-vision-int8.onnx"
VISION_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
    "d15189d7028b43f1d3e65039190477f6af591c2a/onnx/vision_model_quantized.onnx"
)
VISION_SHA256 = "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"
VISION_BYTES = 89_117_001
EMBEDDINGS_FILENAME = "general_label_embeddings.npz"
BACKEND_ID = "clip-vit-b32-int8-583fd111"
LOGIT_SCALE = 100.0
# Ambang dipilih pada foto sungguhan: orang, rumah, dan batu yang jelas lolos,
# sementara noise, warna polos, dan dua label yang hampir seri ditandai tidak yakin.
NULL_MARGIN = 0.022
MIN_LABEL_GAP = 0.006
INPUT_SIZE = 224
_MAX_SIDE = 768
_MAX_ASPECT = 3
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_DOWNLOAD_CHUNK = 1024 * 1024


class GeneralModelError(RuntimeError):
    """Model Umum tidak bisa diunduh, dibaca, atau dijalankan."""


def embeddings_path() -> Path:
    return Path(__file__).resolve().parent / "data" / EMBEDDINGS_FILENAME


def cache_dir() -> Path:
    override = os.environ.get("GENERAL_MODEL_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "cifar10-app"


def vision_model_path() -> Path:
    return cache_dir() / VISION_FILENAME


def vision_weights_cached() -> bool:
    path = vision_model_path()
    return path.is_file() and path.stat().st_size == VISION_BYTES


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "cifar10-app/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as handle:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        dest.unlink(missing_ok=True)
        raise GeneralModelError(
            "Model umum tidak dapat diunduh. Periksa koneksi, lalu muat ulang halaman."
        ) from exc


def ensure_vision_model() -> Path:
    """Pakai berkas cache jika hash-nya cocok. Jika tidak, unduh lalu periksa."""
    path = vision_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and _file_sha256(path) == VISION_SHA256:
        return path
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        _download(VISION_URL, temporary)
        if _file_sha256(temporary) != VISION_SHA256:
            raise GeneralModelError(
                "Berkas model umum tidak utuh. Muat ulang halaman untuk mengunduh lagi."
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    rows = np.array(vectors, dtype=np.float32, copy=True)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    rows /= norms
    return rows


def _softmax(scores: np.ndarray) -> np.ndarray:
    logits = LOGIT_SCALE * np.asarray(scores, dtype=np.float64)
    logits -= np.max(logits)
    exponents = np.exp(logits)
    total = float(exponents.sum())
    if not np.isfinite(total) or total <= 0:
        raise GeneralModelError("Skor label tidak valid.")
    return exponents / total


def score_embeddings(
    image_embed: np.ndarray,
    prompt_embeds: np.ndarray,
    prompt_label_index: np.ndarray,
    null_embeds: np.ndarray,
) -> Prediction:
    """Pilih label dari kemiripan cosine. Beberapa prompt untuk satu label diambil yang maksimum."""
    image = _unit_rows(np.asarray(image_embed, dtype=np.float32).reshape(-1))[0]
    prompts = _unit_rows(prompt_embeds)
    nulls = _unit_rows(null_embeds)
    index = np.asarray(prompt_label_index, dtype=np.int32).reshape(-1)
    if prompts.ndim != 2 or prompts.shape[0] != index.shape[0] or prompts.shape[1] != image.shape[0]:
        raise GeneralModelError("Bentuk embedding label tidak sesuai.")
    if nulls.ndim != 2 or nulls.shape[1] != image.shape[0] or nulls.shape[0] == 0:
        raise GeneralModelError("Embedding pembanding tidak sesuai.")
    similarities = prompts @ image
    if not np.all(np.isfinite(similarities)):
        raise GeneralModelError("Embedding gambar tidak valid.")
    label_scores = np.full(len(LABELS), -np.inf, dtype=np.float32)
    np.maximum.at(label_scores, index, similarities)
    if not np.all(np.isfinite(label_scores)):
        raise GeneralModelError("Embedding label tidak lengkap.")
    null_score = float(np.max(nulls @ image))
    order = np.argsort(-label_scores)
    best = float(label_scores[order[0]])
    second = float(label_scores[order[1]]) if order.size > 1 else best
    probabilities = _softmax(label_scores)
    ranking = tuple(
        RankedClass(LABELS[int(item)].name_id, float(probabilities[int(item)]))
        for item in order
    )
    low_confidence = (best - null_score) < NULL_MARGIN or (best - second) < MIN_LABEL_GAP
    top = ranking[0]
    return Prediction(
        label=top.label,
        confidence=top.probability,
        low_confidence=low_confidence,
        ranking=ranking,
    )


def _limit_aspect(image: Image.Image, max_ratio: int) -> Image.Image:
    width, height = image.size
    if width > height * max_ratio:
        new_width = height * max_ratio
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    if height > width * max_ratio:
        new_height = width * max_ratio
        top = (height - new_height) // 2
        return image.crop((0, top, width, top + new_height))
    return image


def preprocess_general(image: Image.Image) -> np.ndarray:
    """Batch NCHW 224x224 yang dinormalisasi seperti praproses CLIP."""
    prepared = image.convert("RGB")
    prepared.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.Resampling.BICUBIC)
    prepared = _limit_aspect(prepared, _MAX_ASPECT)
    width, height = prepared.size
    if width < 1 or height < 1:
        raise GeneralModelError("Gambar terlalu kecil untuk mode Umum.")
    if width < height:
        new_width = INPUT_SIZE
        new_height = max(INPUT_SIZE, int(round(height * INPUT_SIZE / width)))
    else:
        new_height = INPUT_SIZE
        new_width = max(INPUT_SIZE, int(round(width * INPUT_SIZE / height)))
    resized = prepared.resize((new_width, new_height), Image.Resampling.BICUBIC)
    left = max(0, (new_width - INPUT_SIZE) // 2)
    top = max(0, (new_height - INPUT_SIZE) // 2)
    cropped = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    if cropped.size != (INPUT_SIZE, INPUT_SIZE):
        cropped = cropped.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = (array - CLIP_MEAN) / CLIP_STD
    array = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(array[np.newaxis, ...], dtype=np.float32)


def load_embedding_pack(path: Path | None = None) -> dict[str, np.ndarray]:
    source = embeddings_path() if path is None else Path(path)
    if not source.is_file():
        raise GeneralModelError("Berkas label mode Umum tidak ditemukan.")
    with np.load(source, allow_pickle=False) as data:
        embeddings = np.array(data["embeddings"], dtype=np.float32, copy=True)
        label_index = np.array(data["label_index"], dtype=np.int32, copy=True)
        null_embeddings = np.array(data["null_embeddings"], dtype=np.float32, copy=True)
        keys = tuple(str(item) for item in data["keys"].tolist())
    expected = tuple(label.key for label in LABELS)
    if keys != expected:
        raise GeneralModelError("Berkas label umum tidak cocok dengan daftar label.")
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise GeneralModelError("Bentuk embedding label tidak sesuai.")
    if label_index.shape != (embeddings.shape[0],):
        raise GeneralModelError("Indeks label tidak sesuai dengan embedding.")
    if label_index.size == 0 or int(label_index.min()) < 0 or int(label_index.max()) >= len(LABELS):
        raise GeneralModelError("Indeks label di luar daftar.")
    if set(label_index.tolist()) != set(range(len(LABELS))):
        raise GeneralModelError("Ada label tanpa embedding.")
    if null_embeddings.shape != (len(NULL_PROMPTS), 512):
        raise GeneralModelError("Embedding pembanding tidak sesuai.")
    return {
        "embeddings": embeddings,
        "label_index": label_index,
        "null_embeddings": null_embeddings,
    }


class GeneralBackend:
    """Session ONNX vision plus embedding label yang sudah dihitung."""

    def __init__(self, session: object, pack: dict[str, np.ndarray], cache_id: str = BACKEND_ID):
        self._session = session
        self._lock = threading.Lock()
        self._input_name = session.get_inputs()[0].name
        self.prompt_embeds = pack["embeddings"]
        self.prompt_labels = pack["label_index"]
        self.null_embeds = pack["null_embeddings"]
        self.cache_id = cache_id

    def embed_image(self, image: Image.Image) -> np.ndarray:
        batch = preprocess_general(image)
        try:
            with self._lock:
                output = self._session.run(["image_embeds"], {self._input_name: batch})[0]
        except Exception as exc:
            raise GeneralModelError("Prediksi mode Umum gagal untuk gambar ini.") from exc
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def predict(self, image: Image.Image) -> Prediction:
        return score_embeddings(
            self.embed_image(image),
            self.prompt_embeds,
            self.prompt_labels,
            self.null_embeds,
        )


class _FixedBackend:
    """Backend uji. Tidak memuat ONNX."""

    def __init__(self, prediction: Prediction, cache_id: str):
        self._prediction = prediction
        self.cache_id = cache_id

    def predict(self, image: Image.Image) -> Prediction:
        return self._prediction


def _mock_backend(kind: str) -> _FixedBackend:
    if kind == "unsure":
        ranking = (
            RankedClass("Komputer", 0.08),
            RankedClass("Ladang", 0.05),
            RankedClass("Garpu", 0.04),
            RankedClass("Mouse komputer", 0.02),
            RankedClass("Semut", 0.02),
        )
        return _FixedBackend(Prediction("Komputer", 0.08, True, ranking), "mock-unsure")
    if kind == "sure":
        ranking = (
            RankedClass("Rumah", 0.53),
            RankedClass("Pondok", 0.18),
            RankedClass("Kabin", 0.04),
            RankedClass("Lumbung", 0.04),
            RankedClass("Desa", 0.02),
        )
        return _FixedBackend(Prediction("Rumah", 0.53, False, ranking), "mock-sure")
    raise GeneralModelError("Mode uji model umum tidak dikenal.")


def load_backend() -> GeneralBackend | _FixedBackend:
    mock = os.environ.get("GENERAL_MODEL_MOCK")
    if mock:
        return _mock_backend(mock)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise GeneralModelError("Dependensi onnxruntime belum terpasang.") from exc
    model_path = ensure_vision_model()
    pack = load_embedding_pack()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise GeneralModelError(
            "Model umum tidak dapat dimuat. Muat ulang halaman, lalu coba lagi."
        ) from exc
    return GeneralBackend(session, pack)


def english_name(name_id: str) -> str | None:
    label = label_by_id(name_id)
    if label is None:
        return None
    return label.name_en
