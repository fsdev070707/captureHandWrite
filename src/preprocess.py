# preprocess.py
# Image preprocessing utilities: convert PIL images to 28x28 grayscale tensors for the CNN.

from typing import Tuple
import numpy as np
from PIL import Image, ImageOps
from .config import IMAGE_SIZE

def pil_to_mnist_tensor(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a (1, 28, 28, 1) float32 tensor in range [0, 1].
    
    Steps:
    1) Convert to grayscale (L).
    2) Resize to 28x28 with ANTIALIAS for smoother downsampling.
    3) Invert colors so drawn strokes are bright on dark background.
    4) Normalize to 0..1.
    5) Expand to NHWC: (1, 28, 28, 1).
    """
    # Ensure single-channel grayscale
    gray = img.convert("L")
    # Resize to target MNIST resolution
    small = gray.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    # Invert so ink is light and background is dark (MNIST-like)
    inv = ImageOps.invert(small)
    # Convert to numpy and scale
    arr = np.array(inv, dtype=np.float32) / 255.0
    # Add channel dimension and batch dimension: (28,28) -> (1,28,28,1)
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr
