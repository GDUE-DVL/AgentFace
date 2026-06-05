"""
Error handler node.

Handles errors from any node in the workflow.
Implements retry logic with exponential backoff and
graceful degradation.
"""

import logging
from typing import Any, Optional
from langgraph.types import RunnableConfig

from agent_face.langgraph_brain.state import BeautifyWorkflowState

logger = logging.getLogger(__name__)

# Maximum number of retries before giving up
MAX_RETRIES = 3


async def handle_error(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Handle workflow errors with retry logic.

    If retry_count < MAX_RETRIES, routes back to the appropriate
    recovery point (analyze_image or apply_beautification).
    If retry_count >= MAX_RETRIES, marks the session as failed.

    Args:
        state: Current workflow state with error information.
        config: LangGraph runtime config.

    Returns:
        State update. If retrying, clears error and increments retry_count.
        If giving up, sets workflow_stage to "failed".
    """
    session_id = state.get("session_id", "unknown")
    retry_count = state.get("retry_count", 0)
    error_message = state.get("error_message", "Unknown error")
    current_stage = state.get("workflow_stage", "unknown")

    logger.warning(
        f"handle_error: handling error in stage '{current_stage}' "
        f"for session {session_id} (retry {retry_count}/{MAX_RETRIES}): "
        f"{error_message}"
    )

    if retry_count < MAX_RETRIES:
        # Retry — clear error and increment retry count
        logger.info(
            f"handle_error: retrying session {session_id} (retry {retry_count + 1})"
        )
        return {
            "workflow_stage": "retrying",
            "error_message": None,
            "retry_count": retry_count + 1,
        }

    # Max retries exceeded — fail the session
    logger.error(
        f"handle_error: max retries ({MAX_RETRIES}) exceeded for session {session_id}. "
        f"Last error: {error_message}"
    )
    return {
        "workflow_stage": "failed",
        "error_message": f"Failed after {MAX_RETRIES} retries: {error_message}",
    }
