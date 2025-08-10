# config.py
# Centralized project configuration.

from pathlib import Path

# Base directory of the repo (two parents up from this file)
BASE_DIR = Path(__file__).resolve().parents[1]

# Model directory and default model path
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "mnist_cnn.keras"

# Target image size for MNIST (28x28)
IMAGE_SIZE = (28, 28)
