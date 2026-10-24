"""Training ulang model CNN CIFAR-10.

Input model: gambar RGB 32x32 dengan nilai piksel 0-1 (sama seperti preprocess di app.py).
Jalankan: python train.py [--epochs 40] [--output model_cnn_cifar10_v3.h5]
"""
import argparse
import math
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    def conv_block(x, filters, dropout):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              kernel_regularizer=keras.regularizers.l2(5e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        return layers.Dropout(dropout)(x)

    inputs = keras.Input(shape=(32, 32, 3))
    x = conv_block(inputs, 64, 0.2)
    x = conv_block(x, 128, 0.3)
    x = conv_block(x, 256, 0.4)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def make_datasets(batch_size):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.125, 0.125, fill_mode="reflect"),
    ])

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50000)
        .batch(batch_size)
        .map(lambda x, y: (augment(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds


class TimeBudget(keras.callbacks.Callback):
    """Kurangi jumlah epoch agar training selesai dalam batas waktu (mis. job CI)."""

    def __init__(self, state, budget_minutes):
        super().__init__()
        self.state = state
        self.budget = budget_minutes * 60

    def on_train_begin(self, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            per_epoch = time.time() - self.start
            fit = max(3, int(self.budget / per_epoch))
            if fit < self.state["epochs"]:
                print(f"\nBatas waktu: epoch dikurangi {self.state['epochs']} -> {fit}")
                self.state["epochs"] = fit
        if epoch + 1 >= self.state["epochs"]:
            self.model.stop_training = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="model_cnn_cifar10_v3.h5")
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    train_ds, test_ds = make_datasets(args.batch_size)

    # Warmup 2 epoch lalu cosine decay; total epoch bisa dikurangi TimeBudget.
    state = {"epochs": args.epochs}
    peak_lr, warmup = 2e-3, 2

    def lr_at(epoch):
        if epoch < warmup:
            return peak_lr * (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, state["epochs"] - warmup)
        return 0.5 * peak_lr * (1 + math.cos(math.pi * min(1.0, progress)))

    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_at),
        keras.callbacks.ModelCheckpoint(
            args.output, monitor="val_accuracy", save_best_only=True
        ),
    ]
    if args.time_budget_minutes:
        callbacks.append(TimeBudget(state, args.time_budget_minutes))

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(peak_lr, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    best = keras.models.load_model(args.output)
    # Simpan ulang tanpa state optimizer agar file model lebih kecil.
    best.save(args.output, include_optimizer=False)
    _, acc = best.evaluate(test_ds, verbose=0)
    print(f"Test accuracy ({args.output}): {acc:.4f}")


if __name__ == "__main__":
    main()
