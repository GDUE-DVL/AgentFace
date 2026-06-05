"""
Node 5: apply_beautification

BRIDGE node — calls the MAF orchestrator through the bridge client
to apply beautification parameters to the image.
"""

import logging
from typing import Any, Optional
from langgraph.types import RunnableConfig

from agent_face.langgraph_brain.state import BeautifyWorkflowState
from agent_face.bridge.types import BeautificationRequest
from agent_face.bridge.maf_client import BridgeError

logger = logging.getLogger(__name__)


async def apply_beautification(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Apply beautification to the image using the beautification model via MAF bridge.

    This node:
    1. Takes the final merged params and original image
    2. Calls the bridge client's apply_beautification
    3. Stores the beautified image in state

    On failure, routes to handle_error via the conditional edge.
    """
    bridge = (config.get("configurable", {}) if config else {}).get("bridge")
    if bridge is None:
        return {
            "workflow_stage": "error",
            "error_message": "Bridge client not available in config",
        }

    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "unknown")
    final_params = state.get("final_params")

    if not final_params:
        return {
            "workflow_stage": "error",
            "error_message": "No final_params available — plan must be confirmed first",
        }

    logger.info(
        f"apply_beautification: applying beautification for session {session_id}"
    )

    try:
        request = BeautificationRequest(
            image_b64=state["input_image_b64"],
            params=final_params,
        )

        response = await bridge.apply_beautification(
            request=request,
            user_id=user_id,
            session_id=session_id,
        )

        logger.info(
            f"apply_beautification: complete in {response.latency_ms:.0f}ms "
            f"(safety_warnings={len(response.safety_checks.get('warnings', []))})"
        )

        return {
            "workflow_stage": "beautified",
            "beautified_image_b64": response.image_b64,
        }

    except BridgeError as e:
        logger.error(f"apply_beautification: bridge error: {e}")
        return {
            "workflow_stage": "error",
            "error_message": str(e),
            "retry_count": state.get("retry_count", 0) + 1,
        }
    except Exception as e:
        logger.exception(f"apply_beautification: unexpected error: {e}")
        return {
            "workflow_stage": "error",
            "error_message": f"Unexpected error during beautification: {e}",
            "retry_count": state.get("retry_count", 0) + 1,
        }
