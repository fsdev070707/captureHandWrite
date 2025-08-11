import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from .config import DEFAULT_MODEL_PATH

def predict_topk(x, k=3):
    m = load_or_train(DEFAULT_MODEL_PATH)
    probs = m.predict(x, verbose=0)[0]
    idxs = np.argsort(-probs)[:k]
    return [(int(i), float(probs[i])) for i in idxs]

def build_cnn(input_shape=(28,28,1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    m = models.Model(inputs, outputs)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def load_or_train(model_path: Path = DEFAULT_MODEL_PATH, epochs: int = 1, verbose: int = 2):
    if Path(model_path).exists():
        return tf.keras.models.load_model(model_path)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32")/255.0)[...,None]
    x_test  = (x_test.astype("float32")/255.0)[...,None]
    m = build_cnn()
    m.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1, verbose=verbose)

    # ✅ ensure directory exists before saving
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    m.save(model_path)
    return m

def predict_tensor(x: np.ndarray) -> int:
    m = load_or_train(DEFAULT_MODEL_PATH)
    probs = m.predict(x, verbose=0)[0]
    return int(np.argmax(probs))
