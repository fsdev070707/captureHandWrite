# predict.py
# CLI to predict a digit from an image file using the CNN.

import argparse
from PIL import Image
from .preprocess import pil_to_mnist_tensor
from .model import predict_tensor, load_or_train
from .config import DEFAULT_MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description="Predict MNIST digit from an image file.")
    parser.add_argument("--image", required=True, help="Path to image file (PNG/JPG).")
    args = parser.parse_args()

    # Ensure model is available
    load_or_train(DEFAULT_MODEL_PATH, epochs=1, verbose=0)

    # Load and preprocess
    img = Image.open(args.image)
    x = pil_to_mnist_tensor(img)

    # Predict
    pred = predict_tensor(x)
    print(f"Predicted digit: {pred}")

if __name__ == "__main__":
    main()
