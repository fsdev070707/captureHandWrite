import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
from .preprocess import pil_to_mnist_tensor
from .model import predict_tensor, load_or_train
from .config import DEFAULT_MODEL_PATH

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .model import predict_topk


BRUSH_SIZE = 18  # or 20
CANVAS_SIZE = 280
BG_COLOR = "black"
FG_COLOR = "white"


class DrawApp(tk.Tk):
def on_predict(self):
    x = pil_to_mnist_tensor(self.img)
    # guard: ensure correct type/shape and not all zeros
    if x is None or not hasattr(x, "shape") or x.shape != (1, 28, 28, 1):
        self.pred_var.set("Prediction: (no input)")
        return
    if float(x.sum()) < 1e-6:
        self.pred_var.set("Prediction: (draw something)")
        return
    from .model import predict_topk  # if you added it
    try:
        top3 = predict_topk(x, k=3)
        self.pred_var.set("Prediction: " + ", ".join([f"{d} ({p:.2f})" for d,p in top3]))
    except Exception as e:
        self.pred_var.set(f"Error: {e}")


    def __init__(self):
        super().__init__()
        self.title("captureHandWriteTF - Draw")
        self.img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_COLOR)
        self.draw = ImageDraw.Draw(self.img)
        self.canvas = tk.Canvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BG_COLOR, cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        tk.Button(self, text="View 28x28", command=self.view_28x28).grid(row=1, column=5, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="MNIST Samples", command=self.show_mnist).grid(row=1, column=4, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Predict", command=self.on_predict).grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        tk.Button(self, text="Clear", command=self.clear_canvas).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        tk.Button(self, text="Save PNG", command=self.save_png).grid(row=1, column=2, sticky="ew", padx=5, pady=5)
        tk.Button(self, text="(Re)Train/Load", command=self.ensure_model).grid(row=1, column=3, sticky="ew", padx=5, pady=5)
        self.pred_var = tk.StringVar(value="Prediction: -")
        tk.Label(self, textvariable=self.pred_var, font=("Arial", 14)).grid(row=2, column=0, columnspan=4, pady=5)
        self.last_x = self.last_y = None
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.ensure_model()

    def view_28x28(self):
        x = pil_to_mnist_tensor(self.img)[0, ..., 0]
        plt.figure(figsize=(3,3))
        plt.imshow(x, cmap="gray", vmin=0, vmax=1)
        plt.title("Model Input (28x28)")
        plt.axis("off")
        plt.show()

    def show_mnist(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        idxs = np.random.choice(len(x_train), size=9, replace=False)
        plt.figure(figsize=(5,5))
        for i, idx in enumerate(idxs):
            plt.subplot(3,3,i+1)
            plt.imshow(x_train[idx], cmap="gray")
            plt.title(f"Label: {y_train[idx]}")
            plt.axis("off")
        plt.suptitle("Random MNIST Training Samples", fontsize=14)
        plt.tight_layout()
        plt.show()

    def ensure_model(self):
        try:
            load_or_train(DEFAULT_MODEL_PATH, epochs=1, verbose=0)
            messagebox.showinfo("Model", "Model loaded (trained if needed).")
        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    def on_down(self, e):
        self.last_x, self.last_y = e.x, e.y

    def on_move(self, e):
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, e.x, e.y, fill=FG_COLOR, width=BRUSH_SIZE, capstyle=tk.ROUND, smooth=True)
            self.draw.line((self.last_x, self.last_y, e.x, e.y), fill=FG_COLOR, width=BRUSH_SIZE)
        self.last_x, self.last_y = e.x, e.y

    def on_up(self, e):
        self.last_x = self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.img.paste(BG_COLOR, [0,0,CANVAS_SIZE,CANVAS_SIZE])
        self.pred_var.set("Prediction: -")

    def save_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if path:
            self.img.save(path)
            messagebox.showinfo("Saved", f"Saved to {path}")

    def on_predict(self):
        x = pil_to_mnist_tensor(self.img)
        pred = predict_tensor(x)
        self.pred_var.set(f"Prediction: {pred}")

if __name__ == "__main__":
    app = DrawApp()
    app.mainloop()
