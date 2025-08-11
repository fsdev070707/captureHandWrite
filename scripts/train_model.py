# Run as: python -m scripts.train_model --epochs 1  # Script usage example
import argparse  # Import argparse for command-line argument parsing
from src.model import load_or_train  # Import the model training/loading function
from src.config import DEFAULT_MODEL_PATH  # Import the default model path
def main():
    p = argparse.ArgumentParser()  # Create an ArgumentParser object
    p.add_argument("--epochs", type=int, default=1)  # Add an argument for number of epochs
    args = p.parse_args()  # Parse command-line arguments
    load_or_train(DEFAULT_MODEL_PATH, epochs=args.epochs, verbose=2)  # Train or load the model
    print(f"Saved to: {DEFAULT_MODEL_PATH}")  # Print the path where the model is saved

if __name__ == "__main__":  # Check if the script is run directly
    main()  # Call the main function
