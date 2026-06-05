"""
Node 6: collect_feedback

Human-in-the-Loop (HITL) interrupt point 2.
Pauses the workflow and returns the beautified image to the user
for review and feedback.
"""

import logging
from typing import Any, Optional
from langgraph.types import RunnableConfig
from langgraph.types import interrupt

from agent_face.langgraph_brain.state import BeautifyWorkflowState

logger = logging.getLogger(__name__)


async def collect_feedback(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Present the beautified image to the user and collect feedback.

    This node calls LangGraph's `interrupt()` to pause execution.
    The user can:
    - Rate satisfaction (1-5)
    - Provide comments
    - Request parameter adjustments for a re-do

    The user's feedback is received via `Command(resume=...)` from the API.

    Note: If the user requests a re-do with adjustments, the workflow
    routes back to apply_beautification.
    """
    session_id = state.get("session_id", "unknown")
    beautified_image = state.get("beautified_image_b64")

    if not beautified_image:
        return {
            "workflow_stage": "error",
            "error_message": "No beautified image available to present",
        }

    logger.info(f"collect_feedback: presenting result for session {session_id}")

    # HITL interrupt — workflow pauses here
    # The payload includes the beautified image for the client to display
    feedback_payload = {
        "beautified_image_b64": beautified_image,
        "final_params": state.get("final_params"),
        "message": "Please review the beautified image and provide feedback (1-5).",
    }

    user_feedback = interrupt(feedback_payload)

    # Process feedback
    satisfaction = user_feedback.get("satisfaction_score", 0)

    logger.info(
        f"collect_feedback: received feedback for session {session_id}: "
        f"satisfaction={satisfaction}"
    )

    return {
        "workflow_stage": "feedback_collected",
        "user_feedback": {
            "satisfaction_score": satisfaction,
            "param_adjustments": user_feedback.get("param_adjustments"),
            "comments": user_feedback.get("comments"),
        },
    }
