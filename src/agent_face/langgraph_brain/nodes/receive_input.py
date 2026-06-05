"""
Node 1: receive_input

Validates the input image (format, size, base64 normalization)
and initializes the workflow state.
"""

import base64
import io
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from langgraph.types import RunnableConfig

from PIL import Image
from agent_face.langgraph_brain.state import BeautifyWorkflowState
from agent_face.config import settings

logger = logging.getLogger(__name__)


async def receive_input(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Validate and normalize the input image.

    Checks:
    - base64 encoding validity
    - Image format and size constraints
    - Sets up session metadata

    Returns partial state updates.
    """
    image_b64 = state.get("input_image_b64", "")
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "unknown")

    logger.info(f"receive_input: validating image for session {session_id}")

    errors = []

    # 1. Validate base64
    try:
        decoded = base64.b64decode(image_b64)
    except Exception as e:
        errors.append(f"Invalid base64 encoding: {e}")
        return {
            "workflow_stage": "input_failed",
            "error_message": "; ".join(errors),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # 2. Validate size
    if len(decoded) > settings.max_image_bytes:
        errors.append(
            f"Image too large: {len(decoded)} bytes (max {settings.max_image_bytes})"
        )

    # 3. Validate image format
    try:
        image = Image.open(io.BytesIO(decoded))
        image.verify()
    except Exception as e:
        errors.append(f"Invalid image format: {e}")

    if errors:
        return {
            "workflow_stage": "input_failed",
            "error_message": "; ".join(errors),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # 4. Normalize base64 (strip data URI prefix if present)
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[-1]

    # 6. Initialize session metadata if not set
    created_at = state.get("created_at") or datetime.now(timezone.utc).isoformat()

    logger.info(f"receive_input: ready for session {session_id} ({len(image_b64)} chars base64)")

    return {
        "input_image_b64": image_b64,
        "created_at": created_at,
        "workflow_stage": "input_received",
        "retry_count": 0,
        "messages": [],
    }
