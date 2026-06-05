"""
Beautification client — calls real beauty model via HTTP.

Model API: POST /beautify (multipart form)
  image: file
  prompt: text guidance
  model: deepfrr | ffhqr
  steps: 50
  ...
Returns: binary image
"""

import base64
import io
import logging
import time

import httpx
from PIL import Image

from agent_face.config import settings
from agent_face.langgraph_brain.state import BeautifyParams, BEAUTIFY_PARAM_LABELS

logger = logging.getLogger(__name__)

MAX_SIZE = 512  # Downsample before sending to model


class BeautifyModelClient:
    """Calls the real beautification model API."""

    def __init__(self):
        self._base_url = settings.beautify_model_url.rstrip("/")
        self._timeout = settings.model_request_timeout
        self._model_name = settings.beauty_model_name
        self._steps = settings.beauty_steps
        self._guidance = settings.beauty_guidance_scale
        self._image_guidance = settings.beauty_image_guidance_scale
        self._seed = settings.beauty_seed

    # English labels for beauty model prompt
    PARAM_LABELS_EN = {
        "skin_smoothing": "skin smoothing",
        "whitening": "skin brightening",
        "eye_enlargement": "eye enlarging",
        "face_slimming": "face slimming",
        "blush": "blush",
        "lip_color_adjustment": "lip color adjustment",
        "blemish_removal": "blemish removal",
        "nose_reshaping": "nose reshaping",
        "eyebrow_adjustment": "eyebrow adjustment",
    }

    @staticmethod
    def params_to_description(params: BeautifyParams) -> str:
        """Convert beautification parameters to English prompt with levels."""
        if not any(params.get(k, 0) > 0 for k in BeautifyModelClient.PARAM_LABELS_EN):
            return "natural beauty, clean skin, professional photo"

        # Main beauty instruction with specific levels
        lines = ["Beautify this portrait photo."]
        lines.append("Apply the following adjustments at the specified intensity levels (0-5 scale):")

        for key, en_label in BeautifyModelClient.PARAM_LABELS_EN.items():
            value = params.get(key, 0)
            if value > 0:
                if value <= 1.0:
                    level = "very light"
                elif value <= 2.0:
                    level = "light"
                elif value <= 3.0:
                    level = "moderate"
                elif value <= 4.0:
                    level = "strong"
                else:
                    level = "very strong"
                lines.append(f"  - {en_label}: {level} (level {value:.1f}/5.0)")

        # Style guidance
        lines.append("")
        lines.append("Style guidance: natural look, professional photo quality, preserve facial identity and features.")

        return "\n".join(lines)

    @staticmethod
    def _downsample(image_b64: str) -> bytes:
        """Downsample image to MAX_SIZE and return JPEG bytes."""
        decoded = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(decoded))
        w, h = image.size
        max_side = max(w, h)
        if max_side > MAX_SIZE:
            scale = MAX_SIZE / max_side
            image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            logger.info(f"beautify: downsampled {w}x{h} → {image.size[0]}x{image.size[1]}")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    async def beautify(self, image_b64: str, params: BeautifyParams) -> str:
        """
        Call the beautification model API.

        1. Downsample to 512px
        2. Generate natural language prompt from params
        3. POST to beauty model
        4. Return base64-encoded result
        """
        t0 = time.monotonic()
        prompt = self.params_to_description(params)
        image_bytes = self._downsample(image_b64)

        logger.info(f"beautify: sending to {self._base_url}, prompt={prompt[:80]}...")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/beautify",
                files={"image": ("face.jpg", image_bytes, "image/jpeg")},
                data={
                    "prompt": prompt,
                    "model": self._model_name,
                    "steps": str(self._steps),
                    "guidance_scale": str(self._guidance),
                    "image_guidance_scale": str(self._image_guidance),
                    "seed": str(self._seed),
                },
            )
            resp.raise_for_status()

        # Response is binary image
        result_b64 = base64.b64encode(resp.content).decode()
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"beautify: done in {elapsed:.0f}ms, output={len(result_b64)} chars base64")
        return result_b64

    async def health_check(self) -> dict:
        """Check beauty model health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return {"status": "ok", "latency_ms": resp.elapsed.total_seconds() * 1000}
        except Exception as e:
            return {"status": "error", "error": str(e)}
