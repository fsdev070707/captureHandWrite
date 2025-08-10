# model.py
# Build, train, load, and predict with a small CNN for MNIST digits.

from pathlib import Path
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from .config import DEFAULT_MODEL_PATH

def build_cnn(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    """Build a compact CNN suitable for MNIST digits."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def load_or_train(model_path: Path = DEFAULT_MODEL_PATH, epochs: int = 2, verbose: int = 2) -> tf.keras.Model:
    """Load a saved model if present; otherwise train quickly on MNIST and save it."""
    if Path(model_path).exists():
        return tf.keras.models.load_model(model_path)
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize and add channel dimension
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test  = (x_test.astype("float32") / 255.0)[..., None]
    # Build and train model
    model = build_cnn()
    model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1, verbose=verbose)
    # Evaluate and print brief metrics
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[INFO] Test accuracy: {acc:.4f}")
    # Ensure dir + save
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model

def predict_tensor(x: np.ndarray) -> int:
    """Predict a single (1,28,28,1) tensor and return the integer digit."""
    model = load_or_train(DEFAULT_MODEL_PATH)
    probs = model.predict(x, verbose=0)[0]  # shape (10,)
    pred = int(np.argmax(probs))
    return pred
