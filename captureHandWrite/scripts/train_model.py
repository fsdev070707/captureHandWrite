# Run as: python -m scripts.train_model --epochs 1
import argparse
from src.model import load_or_train
from src.config import DEFAULT_MODEL_PATH

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    load_or_train(DEFAULT_MODEL_PATH, epochs=args.epochs, verbose=2)
    print(f"Saved to: {DEFAULT_MODEL_PATH}")

if __name__ == "__main__":
    main()
