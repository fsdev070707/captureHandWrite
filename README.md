# captureHandWriteTF (fixed)

TensorFlow/Keras CNN for MNIST + Tkinter draw window.
Run from the **project root** so imports like `from src.model import ...` work.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train (from project root)
python -m scripts.train_model --epochs 1

# Launch the draw window
python -m src.draw_app
```
