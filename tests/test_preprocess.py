# tests/test_preprocess.py
# Smoke test for preprocessing: output shape/range should match MNIST expectations.

import numpy as np
from PIL import Image, ImageDraw
from src.preprocess import pil_to_mnist_tensor

def test_pil_to_mnist_tensor_shape_and_range():
    img = Image.new("L", (200, 200), color=0)
    d = ImageDraw.Draw(img)
    d.text((80, 60), "5", fill=255)
    x = pil_to_mnist_tensor(img)
    assert x.shape == (1, 28, 28, 1)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
