"""Hosted OCR (OCR.space) used when the app runs on Vercel.

TrOCR needs torch + transformers (over 1 GB), which cannot fit in a Vercel
serverless function. This module sends the image to the OCR.space web API
instead and returns the text. It only needs Pillow + requests.

The API key is read from the OCR_SPACE_API_KEY environment variable, so it
is never stored in the code or in GitHub.
"""

import io
import os

import requests
from PIL import Image

from image_utils import load_normalized

OCR_SPACE_URL = "https://api.ocr.space/parse/image"

# The free OCR.space plan rejects files larger than 1 MB. Stay a bit under.
MAX_UPLOAD_BYTES = 950 * 1024
REQUEST_TIMEOUT = 40  # seconds (the Vercel function limit is 60)


def cloud_ocr_available():
    """True when an OCR.space API key has been configured."""
    return bool(os.environ.get("OCR_SPACE_API_KEY"))


def _shrink_to_jpeg(image_bytes):
    """Normalise (HEIC, EXIF rotation) and compress to a JPEG under ~1 MB."""
    img = load_normalized(image_bytes)
    for max_side in (2000, 1600, 1280, 1024, 800):
        candidate = img
        if max(candidate.size) > max_side:
            scale = max_side / max(candidate.size)
            candidate = candidate.resize(
                (max(1, round(candidate.width * scale)),
                 max(1, round(candidate.height * scale))),
                Image.LANCZOS,
            )
        for quality in (88, 75, 60):
            buf = io.BytesIO()
            candidate.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= MAX_UPLOAD_BYTES:
                return buf.getvalue()
    return buf.getvalue()  # smallest attempt; let the API report if too big


def extract_text(image_bytes):
    """Send the image to OCR.space and return the recognised text."""
    api_key = os.environ.get("OCR_SPACE_API_KEY")
    if not api_key:
        raise RuntimeError("OCR_SPACE_API_KEY is not set.")

    jpeg_bytes = _shrink_to_jpeg(image_bytes)

    response = requests.post(
        OCR_SPACE_URL,
        headers={"apikey": api_key},
        files={"file": ("prescription.jpg", jpeg_bytes, "image/jpeg")},
        data={
            "language": "eng",
            "OCREngine": "2",       # better with handwriting-like text and digits
            "scale": "true",
            "isTable": "false",
            "isOverlayRequired": "false",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()

    if result.get("IsErroredOnProcessing"):
        message = result.get("ErrorMessage") or result.get("ErrorDetails") or "unknown error"
        if isinstance(message, list):
            message = "; ".join(str(m) for m in message)
        raise RuntimeError(f"OCR.space error: {message}")

    pages = result.get("ParsedResults") or []
    text = "\n".join((p.get("ParsedText") or "").strip() for p in pages)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("OCR.space found no readable text in the image.")
    return "\n".join(lines)
