# captureHandWriteTF (TensorFlow + Draw Window)

An educational, fully-commented project that recognizes **handwritten digits** using a **TensorFlow/Keras CNN**
trained on **MNIST (28×28)**. It includes a **Tkinter draw window** so you can draw digits with your mouse and get
instant predictions.

## Features
- TensorFlow/Keras CNN model (`Conv2D` → `MaxPool` → `Dense`)
- Tkinter-powered drawing canvas (left-drag to draw, buttons to **Predict**, **Clear**, **Save**)
- Image preprocessing to 28×28 grayscale (white ink on black background)
- CLI scripts to train, predict, and launch the draw app
- Fully annotated source files for learning

## Quick start
```bash
# 1) Create a virtual env (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install dependencies (CPU version of TF)
pip install -r requirements.txt

# 3) Train the CNN on MNIST (a small model trains in ~1–2 minutes on CPU)
python scripts/train_model.py --epochs 2

# 4) Launch the draw window and predict
python src/draw_app.py

# 5) (Optional) Predict from an image file (PNG/JPG)
python src/predict.py --image path/to/your_digit.png
```

## Project structure
```
captureHandWriteTF/
  README.md
  requirements.txt
  models/
    mnist_cnn.keras             # saved Keras model (after training)
  scripts/
    train_model.py              # trains/exports the CNN model
  src/
    __init__.py
    config.py                   # central paths + constants (28x28)
    preprocess.py               # PIL image → 28x28 tensor (0..1)
    model.py                    # build/load/predict helpers for CNN
    draw_app.py                 # Tkinter draw window with Predict/Clear/Save
    predict.py                  # CLI: predict from an image path
  tests/
    test_preprocess.py          # smoke test for preprocessing shape/range
```

## Notes
- The draw canvas uses white strokes on black background, then normalizes to 0..1 and resizes to 28×28.
- If your strokes are thin, try drawing thicker or increasing brush size in the code.
- For letters/words OCR, the network and dataset must be extended (happy to scaffold that if you want).


**Build note:** Could not pretrain model here: No module named 'tensorflow'. Run scripts/train_model.py locally.
