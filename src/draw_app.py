# draw_app.py
# Tkinter draw window to input a handwritten digit and predict with the CNN.

import io
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import numpy as np

from .preprocess import pil_to_mnist_tensor
from .model import predict_tensor, load_or_train
from .config import DEFAULT_MODEL_PATH

BRUSH_SIZE = 12      # thickness of the stroke
CANVAS_SIZE = 280    # canvas is 280x280; later resized to 28x28
BG_COLOR = "black"
FG_COLOR = "white"

class DrawApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("captureHandWriteTF - Draw a digit")
        # Create a PIL image buffer to mirror the canvas drawing (for clean preprocessing)
        self.img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), color=BG_COLOR)
        self.draw = ImageDraw.Draw(self.img)

        # Canvas widget for drawing
        self.canvas = tk.Canvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BG_COLOR, cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Buttons: Predict / Clear / Save / Load Model
        tk.Button(self, text="Predict", command=self.on_predict).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Clear", command=self.clear_canvas).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Save PNG", command=self.save_png).grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="(Re)Train/Load Model", command=self.ensure_model).grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        # Label to display prediction
        self.pred_var = tk.StringVar(value="Prediction: -")
        tk.Label(self, textvariable=self.pred_var, font=("Arial", 14)).grid(row=2, column=0, columnspan=4, pady=5)

        # Bind mouse events for drawing
        self.last_x, self.last_y = None, None
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Make sure a model exists (quick load/train if needed)
        self.ensure_model()

    def ensure_model(self):
        try:
            load_or_train(DEFAULT_MODEL_PATH, epochs=1, verbose=0)
            messagebox.showinfo("Model", "Model loaded (trained if not present).")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load/train model: {e}")

    def on_button_press(self, event):
        self.last_x, self.last_y = event.x, event.y

    def on_paint(self, event):
        # Draw on the Tkinter canvas (visual) and on the PIL image (for processing)
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill=FG_COLOR, width=BRUSH_SIZE, capstyle=tk.ROUND, smooth=True)
            self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=FG_COLOR, width=BRUSH_SIZE)
        self.last_x, self.last_y = event.x, event.y

    def on_button_release(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.img.paste(BG_COLOR, [0, 0, CANVAS_SIZE, CANVAS_SIZE])
        self.pred_var.set("Prediction: -")

    def save_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files","*.png")])
        if path:
            self.img.save(path)
            messagebox.showinfo("Saved", f"Saved to {path}")

    def on_predict(self):
        try:
            # Convert the current PIL image (RGB) to model tensor
            x = pil_to_mnist_tensor(self.img)
            pred = predict_tensor(x)
            self.pred_var.set(f"Prediction: {pred}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

if __name__ == "__main__":
    app = DrawApp()
    app.mainloop()
