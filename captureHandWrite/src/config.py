from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "mnist_cnn.keras"
IMAGE_SIZE = (28, 28)
