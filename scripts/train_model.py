# scripts/train_model.py
# Train and save the TensorFlow/Keras CNN on MNIST.

import argparse
from src.model import load_or_train
from src.config import DEFAULT_MODEL_PATH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    args = parser.parse_args()
    model = load_or_train(DEFAULT_MODEL_PATH, epochs=args.epochs, verbose=2)
    print(f"Saved model to: {DEFAULT_MODEL_PATH}")

if __name__ == "__main__":
    main()
