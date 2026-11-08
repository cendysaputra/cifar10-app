"""Latih ResNet-18 CIFAR-10 dan ekspor ke Keras.

Kontrak masukan sama dengan aplikasi: gambar RGB 32x32, piksel 0-1.
Normalisasi rata-rata/simpangan baku ada di dalam model, jadi app.py tidak berubah.

Resep mengikuti DeVries & Taylor (Cutout, 2017) untuk ResNet-18:
pad 4 + crop acak, flip horizontal, Cutout 16, SGD Nesterov, weight decay 5e-4,
200 epoch, batch 128, learning rate 0.1 yang turun 5x setelah epoch 60, 120, dan 160.
Himpunan uji 10.000 gambar tidak dipakai untuk pelatihan atau pemilihan checkpoint.
Validasi diambil dari himpunan latih (500 gambar per kelas).

Mesin ini tidak punya GPU. Konvolusi dilatih dengan bfloat16 pada CPU; bobot dan
evaluasi tetap float32 agar sama dengan inferensi TensorFlow di aplikasi.

Jalankan: python train.py [--epochs 200] [--output model_cnn_cifar10_v5.h5]
"""
from __future__ import annotations

import argparse
import os
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Urutan kelas CIFAR-10 resmi, sama dengan CLASS_NAMES di classify.py.
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

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
VAL_PER_CLASS = 500


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + residual)


class CifarResNet18(nn.Module):
    """ResNet-18 versi CIFAR: stem 3x3, tanpa max-pool 7x7 ImageNet."""

    def __init__(self):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1), BasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def learning_rate(epoch: int) -> float:
    """0-indeks. Turun 5x setelah epoch ke-60, 120, dan 160 selesai."""
    lr = 0.1
    if epoch >= 160:
        return lr * (0.2**3)
    if epoch >= 120:
        return lr * (0.2**2)
    if epoch >= 60:
        return lr * 0.2
    return lr


def download_cifar(root: Path) -> Path:
    extracted = root / "cifar-10-batches-py"
    if extracted.is_dir():
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.is_file():
        print(f"Mengunduh CIFAR-10 dari {CIFAR_URL}")
        urllib.request.urlretrieve(CIFAR_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    return extracted


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        entry = pickle.load(handle, encoding="bytes")
    images = entry[b"data"].reshape(-1, 3, 32, 32)
    labels = np.asarray(entry[b"labels"], dtype=np.int64)
    return images, labels


def load_train_arrays(root: Path) -> tuple[np.ndarray, np.ndarray]:
    folder = download_cifar(root)
    images, labels = [], []
    for index in range(1, 6):
        batch_x, batch_y = _load_batch(folder / f"data_batch_{index}")
        images.append(batch_x)
        labels.append(batch_y)
    return np.concatenate(images), np.concatenate(labels)


def load_test_arrays(root: Path) -> tuple[np.ndarray, np.ndarray]:
    folder = download_cifar(root)
    return _load_batch(folder / "test_batch")


def stratified_split(labels: np.ndarray, val_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for class_id in range(10):
        index = np.flatnonzero(labels == class_id)
        rng.shuffle(index)
        val_parts.append(index[:val_per_class])
        train_parts.append(index[val_per_class:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def channel_mean_std(images_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = images_uint8.astype(np.float64) / 255.0
    mean = pixels.mean(axis=(0, 2, 3))
    std = pixels.std(axis=(0, 2, 3))
    return mean.astype(np.float32), std.astype(np.float32)


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Pad 4, crop 32, flip horizontal. images: NCHW float 0-1."""
    padded = F.pad(images, (4, 4, 4, 4))
    count = images.shape[0]
    out = torch.empty_like(images)
    tops = torch.randint(0, 9, (count,))
    lefts = torch.randint(0, 9, (count,))
    for index in range(count):
        top = int(tops[index])
        left = int(lefts[index])
        out[index] = padded[index, :, top : top + 32, left : left + 32]
    flip = torch.rand(count) < 0.5
    if bool(flip.any()):
        out[flip] = out[flip].flip(-1)
    return out


def apply_cutout(images: torch.Tensor, size: int = 16) -> torch.Tensor:
    """Lubang size x size. Pusat boleh di tepi sehingga sebagian lubang terpotong."""
    count = images.shape[0]
    half = size // 2
    centers_y = torch.randint(0, 32, (count,))
    centers_x = torch.randint(0, 32, (count,))
    for index in range(count):
        y1 = max(0, int(centers_y[index]) - half)
        y2 = min(32, int(centers_y[index]) + half)
        x1 = max(0, int(centers_x[index]) - half)
        x2 = min(32, int(centers_x[index]) + half)
        images[index, :, y1:y2, x1:x2] = 0
    return images


def normalize(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (images - mean) / std


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    decay, plain = [], []
    batch_norm = {
        id(param)
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
        for param in module.parameters()
    }
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or id(param) in batch_norm:
            plain.append(param)
        else:
            decay.append(param)
    return torch.optim.SGD(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=0.1,
        momentum=0.9,
        nesterov=True,
    )


def run_epoch(model, optimizer, images, labels, mean, std, batch_size: int):
    model.train()
    order = torch.randperm(images.shape[0])
    total_loss = 0.0
    correct = 0
    seen = 0
    criterion = nn.CrossEntropyLoss()
    for start in range(0, images.shape[0], batch_size):
        index = order[start : start + batch_size]
        batch = augment_batch(images[index].float().div(255.0))
        batch = normalize(batch, mean, std)
        batch = apply_cutout(batch)
        batch = batch.contiguous(memory_format=torch.channels_last)
        target = labels[index]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(batch)
        loss = criterion(logits.float(), target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * target.shape[0]
        correct += int((logits.float().argmax(dim=1) == target).sum())
        seen += int(target.shape[0])
    return total_loss / seen, correct / seen


@torch.inference_mode()
def evaluate(model, images, labels, mean, std, batch_size: int):
    model.eval()
    correct = 0
    correct_tta = 0
    total = images.shape[0]
    for start in range(0, total, batch_size):
        batch = normalize(images[start : start + batch_size].float().div(255.0), mean, std)
        batch = batch.contiguous(memory_format=torch.channels_last)
        flipped = batch.flip(-1).contiguous(memory_format=torch.channels_last)
        target = labels[start : start + batch_size]
        probs = torch.softmax(model(batch).float(), dim=1)
        probs_flip = torch.softmax(model(flipped).float(), dim=1)
        correct += int((probs.argmax(dim=1) == target).sum())
        correct_tta += int((((probs + probs_flip) * 0.5).argmax(dim=1) == target).sum())
    return correct / total, correct_tta / total


def build_keras_model(mean: np.ndarray, std: np.ndarray):
    from tensorflow import keras
    from tensorflow.keras import layers

    def block(x, filters, strides, prefix):
        shortcut = x
        # Stride 2 memakai padding simetris seperti PyTorch (1 piksel tiap sisi).
        # "same" di Keras menaruh padding hanya di kanan/bawah, jadi fitur bergeser.
        if strides == 1:
            y = layers.Conv2D(
                filters, 3, strides=1, padding="same", use_bias=False, name=f"{prefix}_conv1"
            )(x)
        else:
            y = layers.ZeroPadding2D(padding=(1, 1), name=f"{prefix}_pad")(x)
            y = layers.Conv2D(
                filters, 3, strides=2, padding="valid", use_bias=False, name=f"{prefix}_conv1"
            )(y)
        y = layers.BatchNormalization(epsilon=1e-5, name=f"{prefix}_bn1")(y)
        y = layers.ReLU(name=f"{prefix}_relu1")(y)
        y = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"{prefix}_conv2"
        )(y)
        y = layers.BatchNormalization(epsilon=1e-5, name=f"{prefix}_bn2")(y)
        if strides != 1 or int(shortcut.shape[-1]) != filters:
            shortcut = layers.Conv2D(
                filters, 1, strides=strides, padding="valid", use_bias=False, name=f"{prefix}_proj"
            )(shortcut)
            shortcut = layers.BatchNormalization(epsilon=1e-5, name=f"{prefix}_proj_bn")(shortcut)
        return layers.ReLU(name=f"{prefix}_out")(layers.Add(name=f"{prefix}_add")([y, shortcut]))

    inputs = keras.Input(shape=(32, 32, 3), name="image")
    x = layers.Normalization(
        axis=-1,
        mean=mean.astype(np.float32).tolist(),
        variance=np.square(std.astype(np.float32)).tolist(),
        name="standardize",
    )(inputs)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-5, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    for stage, (filters, stride, count) in enumerate(
        [(64, 1, 2), (128, 2, 2), (256, 2, 2), (512, 2, 2)], start=1
    ):
        for index in range(count):
            x = block(x, filters, stride if index == 0 else 1, f"s{stage}b{index}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="cifar_resnet18")


def _assign_conv(layer, weight: torch.Tensor) -> None:
    layer.kernel.assign(np.ascontiguousarray(weight.detach().float().numpy().transpose(2, 3, 1, 0)))


def _assign_bn(layer, module: nn.BatchNorm2d) -> None:
    layer.gamma.assign(module.weight.detach().float().numpy())
    layer.beta.assign(module.bias.detach().float().numpy())
    layer.moving_mean.assign(module.running_mean.detach().float().numpy())
    layer.moving_variance.assign(module.running_var.detach().float().numpy())


def export_keras(model: CifarResNet18, mean: np.ndarray, std: np.ndarray, path: Path) -> None:
    keras_model = build_keras_model(mean, std)
    _assign_conv(keras_model.get_layer("stem_conv"), model.stem_conv.weight)
    _assign_bn(keras_model.get_layer("stem_bn"), model.stem_bn)
    for stage_index, stage in enumerate(
        (model.layer1, model.layer2, model.layer3, model.layer4), start=1
    ):
        for block_index, block in enumerate(stage):
            prefix = f"s{stage_index}b{block_index}"
            _assign_conv(keras_model.get_layer(f"{prefix}_conv1"), block.conv1.weight)
            _assign_bn(keras_model.get_layer(f"{prefix}_bn1"), block.bn1)
            _assign_conv(keras_model.get_layer(f"{prefix}_conv2"), block.conv2.weight)
            _assign_bn(keras_model.get_layer(f"{prefix}_bn2"), block.bn2)
            if block.shortcut is not None:
                _assign_conv(keras_model.get_layer(f"{prefix}_proj"), block.shortcut[0].weight)
                _assign_bn(keras_model.get_layer(f"{prefix}_proj_bn"), block.shortcut[1])
    dense = keras_model.get_layer("predictions")
    dense.kernel.assign(model.fc.weight.detach().float().numpy().T)
    dense.bias.assign(model.fc.bias.detach().float().numpy())

    _check_export(model, keras_model, mean, std)
    path.parent.mkdir(parents=True, exist_ok=True)
    keras_model.save(path, include_optimizer=False)
    print(f"Model Keras ditulis ke {path} ({path.stat().st_size / 1e6:.1f} MB)")


def _check_export(model, keras_model, mean, std) -> None:
    rng = np.random.default_rng(0)
    images = rng.random((4, 32, 32, 3), dtype=np.float32)
    batch = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean_t = torch.from_numpy(mean).view(1, 3, 1, 1)
    std_t = torch.from_numpy(std).view(1, 3, 1, 1)
    model.eval()
    with torch.inference_mode():
        probs = torch.softmax(model((batch - mean_t) / std_t).float(), dim=1).numpy()
    keras_probs = np.asarray(keras_model(images, training=False))
    delta = float(np.max(np.abs(probs - keras_probs)))
    if delta > 1e-4:
        raise RuntimeError(f"Ekspor Keras tidak cocok dengan PyTorch (selisih {delta:.3e})")
    print(f"Cek ekspor PyTorch vs Keras: selisih probabilitas maks {delta:.3e}")


def self_check() -> None:
    torch.manual_seed(0)
    model = CifarResNet18().eval().to(memory_format=torch.channels_last)
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    export_keras(model, mean, std, Path("/tmp/cifar_resnet18_selfcheck.h5"))
    print("Self-check ekspor berhasil")


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    images, labels = load_train_arrays(DATA_DIR)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        train_index = checkpoint["train_index"]
        val_index = checkpoint["val_index"]
        mean = checkpoint["mean"]
        std = checkpoint["std"]
        start_epoch = int(checkpoint["epoch"]) + 1
        best_tta = float(checkpoint["best_tta"])
        best_state = checkpoint["best_state"]
    else:
        train_index, val_index = stratified_split(labels, args.val_per_class, args.seed)
        mean, std = channel_mean_std(images[train_index])
        start_epoch = 0
        best_tta = -1.0
        best_state = None
        checkpoint = None

    print(
        f"Train {len(train_index)} gambar, validasi {len(val_index)} gambar. "
        f"Mean {mean.tolist()} std {std.tolist()}"
    )
    train_x = torch.from_numpy(np.ascontiguousarray(images[train_index]))
    train_y = torch.from_numpy(labels[train_index])
    val_x = torch.from_numpy(np.ascontiguousarray(images[val_index]))
    val_y = torch.from_numpy(labels[val_index])
    del images, labels

    model = CifarResNet18().to(memory_format=torch.channels_last)
    optimizer = make_optimizer(model)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    mean_t = torch.from_numpy(mean).view(1, 3, 1, 1)
    std_t = torch.from_numpy(std).view(1, 3, 1, 1)
    params = sum(param.numel() for param in model.parameters())
    print(f"ResNet-18 CIFAR: {params / 1e6:.2f} juta parameter")

    log_path = CHECKPOINT_DIR / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if start_epoch == 0 or not log_path.exists():
        log_path.write_text(
            "epoch,lr,train_loss,train_acc,val_acc,val_tta,seconds,best_tta\n",
            encoding="utf-8",
        )

    started = time.time()
    finished_schedule = start_epoch >= args.epochs
    for epoch in range(start_epoch, args.epochs):
        lr = learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_started = time.time()
        train_loss, train_acc = run_epoch(
            model, optimizer, train_x, train_y, mean_t, std_t, args.batch_size
        )
        val_acc, val_tta = evaluate(
            model, val_x, val_y, mean_t, std_t, args.batch_size * 2
        )
        elapsed = time.time() - epoch_started
        if val_tta > best_tta:
            best_tta = val_tta
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        line = (
            f"epoch={epoch + 1}/{args.epochs} lr={lr:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} val_tta={val_tta:.4f} "
            f"best_tta={best_tta:.4f} sec={elapsed:.1f}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch + 1},{lr:.6f},{train_loss:.6f},{train_acc:.6f},"
                f"{val_acc:.6f},{val_tta:.6f},{elapsed:.2f},{best_tta:.6f}\n"
            )
        save_checkpoint(
            CHECKPOINT_DIR / "last.pt",
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_tta": best_tta,
                "best_state": best_state,
                "mean": mean,
                "std": std,
                "train_index": train_index,
                "val_index": val_index,
                "seed": args.seed,
            },
        )
        if args.time_budget_minutes is not None:
            if time.time() - started > args.time_budget_minutes * 60:
                print("Batas waktu tercapai. Checkpoint terbaik disimpan tanpa evaluasi uji.")
                break
        if epoch + 1 == args.epochs:
            finished_schedule = True

    if best_state is None:
        raise RuntimeError("Tidak ada checkpoint validasi.")
    model.load_state_dict(best_state)
    model.eval()
    export_keras(model, mean, std, Path(args.output))
    best_single, best_tta_again = evaluate(
        model, val_x, val_y, mean_t, std_t, args.batch_size * 2
    )
    print(
        f"Checkpoint validasi terbaik: single-view {best_single:.4f}, "
        f"flip TTA {best_tta_again:.4f}"
    )
    if not finished_schedule:
        return

    test_x_np, test_y_np = load_test_arrays(DATA_DIR)
    test_x = torch.from_numpy(np.ascontiguousarray(test_x_np))
    test_y = torch.from_numpy(test_y_np)
    test_single, test_tta = _detailed_test(
        model, test_x, test_y, mean_t, std_t, args.batch_size * 2
    )
    print(f"Test accuracy single-view: {test_single:.4f}")
    print(f"Test accuracy flip-TTA: {test_tta:.4f}")


@torch.inference_mode()
def _detailed_test(model, images, labels, mean, std, batch_size: int):
    model.eval()
    correct = np.zeros(10, dtype=np.int64)
    correct_tta = np.zeros(10, dtype=np.int64)
    totals = np.zeros(10, dtype=np.int64)
    for start in range(0, images.shape[0], batch_size):
        batch = normalize(images[start : start + batch_size].float().div(255.0), mean, std)
        batch = batch.contiguous(memory_format=torch.channels_last)
        flipped = batch.flip(-1).contiguous(memory_format=torch.channels_last)
        target = labels[start : start + batch_size].numpy()
        probs = torch.softmax(model(batch).float(), dim=1)
        probs_flip = torch.softmax(model(flipped).float(), dim=1)
        pred = probs.argmax(dim=1).numpy()
        pred_tta = ((probs + probs_flip) * 0.5).argmax(dim=1).numpy()
        for class_id in range(10):
            mask = target == class_id
            totals[class_id] += int(mask.sum())
            correct[class_id] += int((pred[mask] == class_id).sum())
            correct_tta[class_id] += int((pred_tta[mask] == class_id).sum())
    print("Akurasi per kelas pada himpunan uji (single-view / flip TTA):")
    for class_id, name in enumerate(CLASS_NAMES):
        single = correct[class_id] / totals[class_id]
        tta = correct_tta[class_id] / totals[class_id]
        print(f"  {name}: {single:.4f} / {tta:.4f}")
    return float(correct.sum() / totals.sum()), float(correct_tta.sum() / totals.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="model_cnn_cifar10_v5.h5")
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    train(args)


if __name__ == "__main__":
    main()
