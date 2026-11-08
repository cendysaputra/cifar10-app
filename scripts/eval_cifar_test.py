"""Skor model CIFAR-10 yang sudah dibekukan pada 10.000 gambar uji resmi.

Himpunan uji tidak dipakai untuk melatih atau memilih checkpoint. Skrip ini hanya
mengukur ulang berkas model. Angka yang dicetak harus sama dengan
results/metrics.json.

Butuh dependensi aplikasi (TensorFlow), bukan PyTorch:

    python scripts/eval_cifar_test.py
    python scripts/eval_cifar_test.py --write-results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASS_NAMES = (
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

# Fakta pelatihan yang tidak bisa dibaca dari berkas .h5. Checkpoint dipilih
# dari validasi sebelum evaluasi uji yang pertama, pada 8 November 2026.
TRAINING_RECORD = {
    "dataset": "CIFAR-10",
    "train_images": 45000,
    "validation_images": 5000,
    "validation_per_class": 500,
    "test_images": 10000,
    "seed": 42,
    "epochs_scheduled": 200,
    "epochs_completed": 200,
    "best_checkpoint_epoch": 196,
    "checkpoint_metric": "validation_accuracy_horizontal_flip_tta",
    "validation_accuracy_single_view": 0.9624,
    "validation_accuracy_flip_tta": 0.9648,
    "test_set_used_for_training": False,
    "test_set_used_for_checkpoint_selection": False,
    "training_test_evaluations": 1,
    "training_time_seconds": 23566.9,
    "date": "2026-11-08",
    "hardware": {
        "gpu": False,
        "cpu_cores": 4,
        "cpu_model": "Intel(R) Xeon(R) Processor",
        "training_compute": "bfloat16",
        "exported_weights": "float32",
    },
    "framework_train": "PyTorch 2.15.0+cpu",
    "framework_export": "TensorFlow 2.21.0 / Keras 3.15.1",
    "app_inference": "horizontal_flip_tta",
}


def ensure_test_batch(root: Path) -> Path:
    folder = root / "cifar-10-batches-py"
    test_batch = folder / "test_batch"
    if test_batch.is_file():
        return test_batch
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.is_file():
        print(f"Mengunduh CIFAR-10 dari {CIFAR_URL}")
        urllib.request.urlretrieve(CIFAR_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    if not test_batch.is_file():
        raise FileNotFoundError(test_batch)
    return test_batch


def load_test(root: Path) -> tuple[np.ndarray, np.ndarray]:
    with ensure_test_batch(root).open("rb") as handle:
        entry = pickle.load(handle, encoding="bytes")
    images = entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(entry[b"labels"], dtype=np.int64)
    if images.shape != (10000, 32, 32, 3) or labels.shape != (10000,):
        raise RuntimeError(f"Bentuk uji tidak terduga: {images.shape} {labels.shape}")
    return images, labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(model, images_uint8: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    single = np.empty(len(images_uint8), dtype=np.int64)
    tta = np.empty(len(images_uint8), dtype=np.int64)
    for start in range(0, len(images_uint8), batch_size):
        batch = images_uint8[start : start + batch_size].astype(np.float32) / 255.0
        flipped = np.ascontiguousarray(batch[:, :, ::-1, :])
        probs = np.asarray(model(batch, training=False), dtype=np.float64)
        probs_flip = np.asarray(model(flipped, training=False), dtype=np.float64)
        stop = start + len(batch)
        single[start:stop] = probs.argmax(axis=1)
        tta[start:stop] = ((probs + probs_flip) * 0.5).argmax(axis=1)
    return single, tta


def confusion(labels: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    return matrix.tolist()


def per_class(matrix: list[list[int]]) -> dict[str, float]:
    scores = {}
    for index, name in enumerate(CLASS_NAMES):
        total = sum(matrix[index])
        scores[name] = matrix[index][index] / total
    return scores


def save_confusion_png(matrix: list[list[int]], path: Path) -> None:
    cell = 86
    left = 108
    top = 78
    width = left + 10 * cell + 20
    height = top + 10 * cell + 36
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((16, 16), "CIFAR-10 test confusion matrix, horizontal-flip TTA", fill="black", font=font)
    draw.text((16, 36), "rows = true class, columns = predicted class, counts out of 1000", fill="black", font=font)
    peak = max(max(row) for row in matrix)
    short = list(CLASS_NAMES)
    for index, name in enumerate(short):
        draw.text((8, top + index * cell + 22), name, fill="black", font=font)
        draw.text((left + index * cell + 4, top - 18), name, fill="black", font=font)
    for row in range(10):
        for col in range(10):
            value = matrix[row][col]
            shade = int(255 - 220 * (value / peak))
            color = (shade, shade, 255) if row == col else (255, shade, shade)
            x0 = left + col * cell
            y0 = top + row * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color, outline="white")
            label = str(value)
            draw.text((x0 + 8, y0 + 28), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def accuracy(labels: np.ndarray, predicted: np.ndarray) -> float:
    return float((labels == predicted).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "model_cnn_cifar10_v5.h5"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    from tensorflow.keras.models import load_model

    model = load_model(model_path, compile=False)
    images, labels = load_test(Path(args.data_dir))
    single, tta = predict(model, images, args.batch_size)
    matrix_single = confusion(labels, single)
    matrix_tta = confusion(labels, tta)
    single_acc = accuracy(labels, single)
    tta_acc = accuracy(labels, tta)
    single_correct = int((labels == single).sum())
    tta_correct = int((labels == tta).sum())
    digest = sha256(model_path)

    print(f"model: {model_path.name}")
    print(f"sha256: {digest}")
    print(f"test single-view: {single_correct}/10000 = {single_acc:.4f}")
    print(f"test flip-TTA: {tta_correct}/10000 = {tta_acc:.4f}")
    print("per class single-view / flip-TTA:")
    single_class = per_class(matrix_single)
    tta_class = per_class(matrix_tta)
    for name in CLASS_NAMES:
        print(f"  {name}: {single_class[name]:.4f} / {tta_class[name]:.4f}")

    if not args.write_results:
        return

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    payload = {
        **TRAINING_RECORD,
        "model_file": model_path.name,
        "model_sha256": digest,
        "model_bytes": model_path.stat().st_size,
        "test_correct_single_view": single_correct,
        "test_correct_flip_tta": tta_correct,
        "test_accuracy_single_view": single_acc,
        "test_accuracy_flip_tta": tta_acc,
        "per_class_test_accuracy_single_view": single_class,
        "per_class_test_accuracy_flip_tta": tta_class,
        "confusion_matrix_single_view": matrix_single,
        "confusion_matrix_flip_tta": matrix_tta,
        "confusion_matrix_png": "results/confusion_matrix_tta.png",
    }
    metrics_path = results / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (results / "model_sha256.txt").write_text(f"{digest}  {model_path.name}\n", encoding="utf-8")
    save_confusion_png(matrix_tta, results / "confusion_matrix_tta.png")
    print(f"wrote {metrics_path}")
    print(f"wrote {results / 'model_sha256.txt'}")
    print(f"wrote {results / 'confusion_matrix_tta.png'}")


if __name__ == "__main__":
    main()
