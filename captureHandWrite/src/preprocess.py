import numpy as np
from PIL import Image, ImageOps
from .config import IMAGE_SIZE  # (28, 28)

def _to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")

def _should_invert(gray: Image.Image, border=4, thresh=0.6):
    w, h = gray.size
    a = np.array(gray, dtype=np.float32) / 255.0
    rim = np.concatenate([a[:border,:].ravel(), a[-border:,:].ravel(), a[:, :border].ravel(), a[:, -border:].ravel()])
    return float(rim.mean()) > thresh  # bright border => invert

def _bbox(a: np.ndarray, thresh=0.05):
    m = a > thresh
    if not m.any():
        return None
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max(), ys.max()

def _centroid(a: np.ndarray):
    h, w = a.shape
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    total = a.sum()
    if total <= 1e-6:
        return (h/2, w/2)
    return ((y_idx * a).sum() / total, (x_idx * a).sum() / total)

def pil_to_mnist_tensor(img: Image.Image) -> np.ndarray:
    # 1) grayscale, auto-invert if background is white
    gray = _to_grayscale(img)
    if _should_invert(gray):
        gray = ImageOps.invert(gray)

    # 2) normalize
    a = np.asarray(gray, dtype=np.float32) / 255.0

    # 3) crop to strokes; if empty, return zeros (1,28,28,1)
    bb = _bbox(a, thresh=0.05)
    if bb is None:
        out = np.zeros(IMAGE_SIZE, dtype=np.float32)
        return out[None, ..., None]

    x0, y0, x1, y1 = bb
    cropped = a[y0:y1+1, x0:x1+1]

    # 4) pad to square
    h, w = cropped.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.float32)
    yo = (size - h) // 2
    xo = (size - w) // 2
    square[yo:yo+h, xo:xo+w] = cropped

    # 5) resize to 20x20
    small20 = Image.fromarray((square * 255).astype(np.uint8), "L").resize((20, 20), Image.Resampling.LANCZOS)
    small20 = np.asarray(small20, dtype=np.float32) / 255.0

    # 6) paste into 28x28 and center by centroid
    canvas = np.zeros(IMAGE_SIZE, dtype=np.float32)
    y0c = (28 - 20) // 2
    x0c = (28 - 20) // 2
    canvas[y0c:y0c+20, x0c:x0c+20] = small20

    cy, cx = _centroid(canvas)
    dy = int(round(13.5 - cy))
    dx = int(round(13.5 - cx))
    shifted = np.zeros_like(canvas)

    ys = slice(max(0, y0c+dy), min(28, y0c+dy+20))
    xs = slice(max(0, x0c+dx), min(28, x0c+dx+20))
    ys_src = slice(max(0, -(y0c+dy)), max(0, -(y0c+dy)) + (ys.stop - ys.start))
    xs_src = slice(max(0, -(x0c+dx)), max(0, -(x0c+dx)) + (xs.stop - xs.start))
    shifted[ys, xs] = small20[ys_src, xs_src]

    # always return (1,28,28,1) ndarray (never None)
    return shifted[None, ..., None]
