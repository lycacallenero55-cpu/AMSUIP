import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


def build_preprocess_layers(image_size: int) -> keras.Sequential:

    return keras.Sequential(
        [
            layers.Resizing(image_size, image_size, interpolation="bilinear"),
            layers.Rescaling(1.0 / 255.0),
        ],
        name="preprocess",
    )


def build_augment_layers() -> keras.Sequential:

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1, fill_mode="nearest"),
            layers.RandomZoom(0.1, fill_mode="nearest"),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augment",
    )


def make_tfdata_from_numpy(
    x: tf.Tensor,
    y: tf.Tensor,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    cache: bool = True,
) -> tf.data.Dataset:

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=tf.shape(x)[0])

    preprocess = build_preprocess_layers(image_size)
    augment_layers = build_augment_layers() if augment else None

    def _map(img, label):
        img = preprocess(img)
        if augment_layers is not None:
            img = augment_layers(img)
        return img, label

    autotune = tf.data.AUTOTUNE
    ds = ds.map(_map, num_parallel_calls=autotune)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(autotune)
    return ds


def split_train_val(
    x: tf.Tensor,
    y: tf.Tensor,
    val_fraction: float = 0.2,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    num = tf.shape(x)[0]
    val_size = tf.cast(tf.math.round(tf.cast(num, tf.float32) * val_fraction), tf.int32)
    val_x = x[:val_size]
    val_y = y[:val_size]
    train_x = x[val_size:]
    train_y = y[val_size:]
    return train_x, train_y, val_x, val_y

