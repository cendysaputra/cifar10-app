"""Bangun ulang embedding label mode Umum.

Bukan bagian dari runtime Streamlit. Butuh onnxruntime, tokenizers, dan numpy.

    pip install onnxruntime tokenizers numpy
    python scripts/build_general_embeddings.py

Model teks dan tokenizer diunduh ke cache. Berkas vision tidak diperlukan di sini.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from general_labels import LABELS, NULL_PROMPTS, prompt_texts  # noqa: E402

TEXT_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
    "d15189d7028b43f1d3e65039190477f6af591c2a/onnx/text_model_quantized.onnx"
)
TEXT_SHA256 = "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a"
TOKENIZER_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
    "d15189d7028b43f1d3e65039190477f6af591c2a/tokenizer.json"
)
TOKENIZER_SHA256 = "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd"
CONTEXT_LENGTH = 77
PAD_ID = 49407
OUTPUT = ROOT / "data" / "general_label_embeddings.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_file(env_name: str, url: str, filename: str, expected_sha: str) -> Path:
    override = os.environ.get(env_name)
    if override:
        path = Path(override)
        if _sha256(path) != expected_sha:
            raise SystemExit(f"Hash tidak cocok untuk {path}")
        return path
    path = Path.home() / ".cache" / "cifar10-app" / filename
    if path.is_file() and _sha256(path) == expected_sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "cifar10-app/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if _sha256(temporary) != expected_sha:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Unduhan {filename} tidak utuh")
    temporary.replace(path)
    return path


def _encode(tokenizer: object, prompts: list[str]) -> np.ndarray:
    rows = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        if len(ids) != CONTEXT_LENGTH:
            raise SystemExit(f"Panjang token bukan {CONTEXT_LENGTH}: {prompt}")
        rows.append(ids)
    return np.asarray(rows, dtype=np.int64)


def _embed(session: object, input_ids: np.ndarray) -> np.ndarray:
    output = session.run(["text_embeds"], {"input_ids": input_ids})[0]
    vectors = np.asarray(output, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise SystemExit("Ada embedding teks yang kosong")
    return vectors / norms


def main() -> None:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    text_path = _cache_file("CLIP_TEXT_ONNX", TEXT_URL, "clip-vit-b32-text-int8.onnx", TEXT_SHA256)
    tokenizer_path = _cache_file(
        "CLIP_TOKENIZER",
        TOKENIZER_URL,
        "clip-vit-b32-tokenizer.json",
        TOKENIZER_SHA256,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=CONTEXT_LENGTH)
    tokenizer.enable_padding(length=CONTEXT_LENGTH, pad_id=PAD_ID)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(text_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    prompt_rows: list[str] = []
    label_index: list[int] = []
    for index, label in enumerate(LABELS):
        texts = prompt_texts(label)
        prompt_rows.extend(texts)
        label_index.extend([index] * len(texts))
    embeddings = _embed(session, _encode(tokenizer, prompt_rows))
    null_embeddings = _embed(session, _encode(tokenizer, list(NULL_PROMPTS)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        embeddings=embeddings,
        label_index=np.asarray(label_index, dtype=np.int32),
        null_embeddings=null_embeddings,
        keys=np.asarray([label.key for label in LABELS]),
    )
    print(f"menulis {OUTPUT} ({embeddings.shape[0]} prompt, {len(LABELS)} label)")


if __name__ == "__main__":
    main()
