"""Model builder for FER2013 emotion classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def _add_conv_block(
    model: tf.keras.Sequential,
    filters: int,
    *,
    kernel_size: tuple[int, int] = (3, 3),
    dropout_rate: float = 0.0,
) -> None:
    """Append a Conv-BN-ReLU block optionally followed by dropout."""
    he_init = tf.keras.initializers.HeNormal()
    model.add(
        layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=he_init,
            use_bias=False,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    if dropout_rate:
        model.add(layers.Dropout(dropout_rate))


def _build_backbone(input_shape_rgb: tuple[int, int, int]) -> tf.keras.Model:
    """Return a MobileNetV2 backbone configured for transfer learning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape_rgb,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    return tf.keras.Model(base_model.input, base_model.output, name="mobilenet_v2_base")


def build_sequential_cnn(
    input_shape: tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7,
) -> tf.keras.Model:
    """Construct and compile a Sequential CNN that fine-tunes MobileNetV2 features."""
    tf.keras.utils.set_random_seed(1337)

    rgb_shape = (96, 96, 3)
    backbone = _build_backbone(rgb_shape)
    he_init = tf.keras.initializers.HeNormal()

    model = tf.keras.Sequential(name="fer2013_transfer_cnn")
    model.add(layers.Input(shape=input_shape))

    # Lightweight convolutional front-end operating on the grayscale input.
    _add_conv_block(model, filters=32, dropout_rate=0.1)
    _add_conv_block(model, filters=32)
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.15))

    _add_conv_block(model, filters=64, dropout_rate=0.15)
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    # Project feature maps down to three channels expected by the backbone.
    model.add(
        layers.Conv2D(
            3,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=he_init,
            use_bias=False,
            name="proj_to_rgb",
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    # Match the backbone resolution.
    model.add(layers.Resizing(rgb_shape[0], rgb_shape[1]))
    model.add(layers.Rescaling(2.0, offset=-1.0))

    # Transfer learning backbone.
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())

    # Classification head.
    model.add(layers.Dropout(0.35))
    model.add(
        layers.Dense(
            256,
            activation="relu",
            kernel_initializer=he_init,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


if __name__ == "__main__":
    build_sequential_cnn()
