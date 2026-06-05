"""
Image utility functions — base64 encoding/decoding, resize, validation.
"""

import base64
import io
from typing import Optional, Tuple
from PIL import Image


def decode_base64_image(image_b64: str) -> Image.Image:
    """
    Decode a base64-encoded image string to a PIL Image.

    Handles data URI prefixes (e.g., "data:image/jpeg;base64,...").

    Args:
        image_b64: Base64-encoded image string.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If the base64 string is invalid.
    """
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[-1]

    try:
        decoded = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(decoded))
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")


def encode_image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """
    Encode a PIL Image to a base64 string.

    Args:
        image: PIL Image object.
        fmt: Output format (JPEG, PNG, WEBP).

    Returns:
        Base64-encoded image string.
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_image(
    image: Image.Image,
    max_dimension: int = 4096,
) -> Image.Image:
    """
    Resize an image proportionally so its longest side <= max_dimension.

    Args:
        image: PIL Image object.
        max_dimension: Maximum pixel dimension.

    Returns:
        Resized PIL Image (or original if already within limits).
    """
    width, height = image.size
    max_side = max(width, height)

    if max_side <= max_dimension:
        return image

    scale = max_dimension / max_side
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def get_image_info(image_b64: str) -> dict:
    """
    Get metadata about a base64-encoded image.

    Args:
        image_b64: Base64-encoded image string.

    Returns:
        Dict with width, height, format, and estimated size.
    """
    image = decode_base64_image(image_b64)
    return {
        "width": image.width,
        "height": image.height,
        "format": image.format or "unknown",
        "mode": image.mode,
        "size_bytes": len(base64.b64decode(image_b64.split(",", 1)[-1])),
    }
