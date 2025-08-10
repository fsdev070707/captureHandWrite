import numpy as np
from PIL import Image, ImageOps
from .config import IMAGE_SIZE

def pil_to_mnist_tensor(img: Image.Image) -> np.ndarray:
    gray = img.convert("L")
    small = gray.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    inv = ImageOps.invert(small)
    arr = (np.array(inv, dtype=np.float32) / 255.0)[..., None]
    arr = np.expand_dims(arr, 0)
    return arr
