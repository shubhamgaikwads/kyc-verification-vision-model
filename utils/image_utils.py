import base64
import os
from typing import Tuple, Optional

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def is_supported_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTS


def guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    # Apply EXIF-based transpose (rotation fix) safely
    img = ImageOps.exif_transpose(img)
    return img


def downscale_if_needed(img: Image.Image, max_side: int = 1600) -> Image.Image:
    """Downscale large images for faster/cheaper VLM inference."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size)


def image_to_data_uri(path: str, max_side: int = 1600) -> str:
    """
    Convert local image to base64 data URI.
    - Fixes EXIF orientation
    - Optionally downscales for speed/cost
    """
    img = load_image(path)
    img = downscale_if_needed(img, max_side=max_side)

    mime = guess_mime(path)
    # Always encode as PNG for predictable results (optional choice)
    # If input is already PNG, keep it as PNG; else convert to PNG bytes.
    if mime != "image/png":
        mime = "image/png"

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def infer_doc_type_from_filename(filename: str) -> str:
    lower = filename.strip().lower()
    normalized = lower.replace("-", " ").replace("_", " ")
    if "passport" in normalized or "pass" in normalized:
        return "passport"
    if "license" in normalized or "dl" in normalized or "driver" in normalized:
        return "drivers_license"
    return "unknown"
