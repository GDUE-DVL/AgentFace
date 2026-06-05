"""
Node 3: analyze_image

BRIDGE node — calls the MAF orchestrator through the bridge client
to analyze the facial image using the multimodal model.
"""

import logging
from typing import Any, Optional
from langgraph.types import RunnableConfig

from agent_face.langgraph_brain.state import BeautifyWorkflowState
from agent_face.bridge.types import AnalysisRequest
from agent_face.bridge.maf_client import BridgeError

logger = logging.getLogger(__name__)


async def analyze_image(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Analyze the image using the multimodal model via MAF bridge.

    This node:
    1. Builds an AnalysisRequest from state
    2. Calls the bridge client
    3. Stores the AnalysisResult in state

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

    logger.info(f"analyze_image: analyzing image for session {session_id}")

    # Load feedback summary to drive preference strength
    session_count = 0
    avg_satisfaction = 0.0
    store = (config.get("configurable", {}) if config else {}).get("store")
    if store:
        try:
            from agent_face.langgraph_brain.memory.feedback import get_user_feedback_summary
            summary = await get_user_feedback_summary(store, user_id)
            session_count = summary.get("total_sessions", 0)
            avg_satisfaction = summary.get("avg_satisfaction", 0.0)
        except Exception:
            pass

    try:
        request = AnalysisRequest(
            image_b64=state["input_image_b64"],
            user_prompt=state.get("user_prompt"),
            user_preferences=state.get("user_preferences"),
            session_count=session_count,
            avg_satisfaction=avg_satisfaction,
        )

        response = await bridge.analyze_image(
            request=request,
            user_id=user_id,
            session_id=session_id,
        )

        logger.info(
            f"analyze_image: analysis complete in {response.latency_ms:.0f}ms "
            f"(confidence={response.result.get('confidence', 0):.2f})"
        )

        return {
            "workflow_stage": "analyzed",
            "analysis_result": response.result,
        }

    except BridgeError as e:
        logger.error(f"analyze_image: bridge error: {e}")
        retry_count = state.get("retry_count", 0) + 1
        return {
            "workflow_stage": "error",
            "error_message": str(e),
            "retry_count": retry_count,
        }
    except Exception as e:
        logger.exception(f"analyze_image: unexpected error: {e}")
        return {
            "workflow_stage": "error",
            "error_message": f"Unexpected error during analysis: {e}",
            "retry_count": state.get("retry_count", 0) + 1,
        }
