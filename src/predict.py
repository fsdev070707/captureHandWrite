import argparse
from PIL import Image
from .preprocess import pil_to_mnist_tensor
from .model import predict_tensor, load_or_train
from .config import DEFAULT_MODEL_PATH

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    args = p.parse_args()
    load_or_train(DEFAULT_MODEL_PATH, epochs=1, verbose=0)
    img = Image.open(args.image)
    x = pil_to_mnist_tensor(img)
    print("Predicted:", predict_tensor(x))

if __name__ == "__main__":
    main()
