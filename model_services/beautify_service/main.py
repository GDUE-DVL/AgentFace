"""
Beautification Model Service — FastAPI stub.

This is a placeholder for the actual beautification model service.
Replace this with the real model inference code.

Expected API:
    POST /beautify — Apply beautification to an image
    GET  /health  — Health check

The AgentFace MAF Body communicates with this service via MCP StreamableHTTP.

Input: image_b64 (base64) + description (natural language)
Output: image_b64 (base64) — the beautified image
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(_env_path)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import base64

app = FastAPI(title="Beautification Model Service", version="0.1.0")


class BeautifyRequest(BaseModel):
    image_b64: str
    description: Optional[str] = None


class BeautifyResponse(BaseModel):
    image_b64: str
    description_applied: str
    processing_time_ms: float = 0.0


@app.post("/beautify")
async def beautify(request: BeautifyRequest) -> BeautifyResponse:
    """
    Apply beautification enhancements to a facial image.

    TODO: Replace with actual model inference.

    The model should:
    1. Load the image from base64
    2. Parse the natural language description
    3. Apply the described beautification effects
    4. Return the enhanced image as base64

    The description format is:
        "请对人像进行以下美颜处理：磨皮中度(程度2.0/5.0)；美白轻度(程度1.0/5.0)；..."
    """
    # Placeholder — replace with real model inference.
    # For now, just echo back the input image.
    return BeautifyResponse(
        image_b64=request.image_b64,
        description_applied=request.description or "",
        processing_time_ms=0.0,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
