"""
Node 8: finalize

Formats the final response payload and marks the session as complete.
"""

import logging
from typing import Any, Optional
from langgraph.types import RunnableConfig
from datetime import datetime, timezone

from agent_face.langgraph_brain.state import BeautifyWorkflowState

logger = logging.getLogger(__name__)


async def finalize(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Finalize the beautification session.

    Formats the complete response including:
    - Original analysis
    - Final parameters used
    - Beautified image
    - User feedback
    - Session metadata

    This node is the terminal node of the happy path.
    """
    session_id = state.get("session_id", "unknown")
    user_id = state.get("user_id", "anonymous")

    logger.info(f"finalize: completing session {session_id} for user {user_id}")

    # The state is fully populated at this point.
    # The API layer will format the final response for the client.

    return {
        "workflow_stage": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
