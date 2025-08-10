from PIL import Image, ImageDraw
from src.preprocess import pil_to_mnist_tensor

def test_shape():
    img = Image.new("L", (200,200), color=0)
    d = ImageDraw.Draw(img)
    d.text((80,60), "3", fill=255)
    x = pil_to_mnist_tensor(img)
    assert x.shape == (1,28,28,1)
