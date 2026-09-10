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
import json
import logging
import os
import time

import httpx
from PIL import Image

from agent_face.config import settings
from agent_face.langgraph_brain.state import BeautifyParams, BEAUTIFY_PARAM_LABELS

logger = logging.getLogger(__name__)

MAX_SIZE = int(os.environ.get("BEAUTIFY_MAX_SIZE", "512"))
SEND_FORMAT = os.environ.get("BEAUTIFY_IMAGE_FORMAT", "JPEG").upper()


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
        if SEND_FORMAT == "PNG":
            image.save(buf, format="PNG")
        else:
            image.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    async def beautify(self, image_b64: str, params: BeautifyParams, src_prompt: str = "", target_prompt: str = "", edit_regions: list[dict] | None = None, seed: int | None = None) -> str:
        """
        Call the beautification model API.

        P2P mode: the model's source/target prompt pair is passed straight
        through as the H-Edit conditions — no manual param-tier mapping.

        1. Downsample to 512px
        2. Use target_description as the main prompt condition
        3. POST to beauty model
        4. Return base64-encoded result
        """
        t0 = time.monotonic()
        # 直接使用模型输出的 P2P 提示词组，不再人工映射档位
        prompt = target_prompt or "clean smooth skin, natural skin texture"
        image_bytes = self._downsample(image_b64)
        upload_name = "face.png" if SEND_FORMAT == "PNG" else "face.jpg"
        upload_type = "image/png" if SEND_FORMAT == "PNG" else "image/jpeg"

        logger.info(f"beautify: sending to {self._base_url}, prompt={prompt[:80]}...")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/beautify",
                files={"image": (upload_name, image_bytes, upload_type)},
                data={
                    "prompt": prompt,
                    "src_prompt": src_prompt,
                    "target_prompt": target_prompt,
                    "edit_regions": json.dumps(edit_regions or [], ensure_ascii=False),
                    "model": self._model_name,
                    "steps": str(self._steps),
                    "guidance_scale": str(self._guidance),
                    "image_guidance_scale": str(self._image_guidance),
                    "seed": str(seed if seed is not None else self._seed),
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
