"""
Session history memory operations.

Stores complete session records for user browsing and retrieval.
Namespace: ("users", "{user_id}", "sessions", "{session_id}")
"""

import logging
from typing import Optional
from datetime import datetime, timezone

from agent_face.langgraph_brain.state import BeautifyWorkflowState

logger = logging.getLogger(__name__)


async def save_session_record(
    store,
    state: BeautifyWorkflowState,
) -> None:
    """
    Save a complete session record to the Store.

    Called by the `update_memory` node after the session completes.

    Args:
        store: LangGraph Store instance.
        state: The final workflow state to persist.
    """
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "unknown")

    namespace = ("users", user_id, "sessions", session_id)

    record = {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": state.get("created_at", datetime.now(timezone.utc).isoformat()),
        "workflow_stage": state.get("workflow_stage", "completed"),
        "final_params": state.get("final_params"),
        "user_feedback": state.get("user_feedback"),
    }

    # Don't store full images in session history — too large.
    # Store parameter summaries instead.
    if state.get("analysis_result"):
        analysis = state["analysis_result"]
        record["analysis_summary"] = {
            "skin_tone": analysis.get("skin_tone"),
            "skin_condition": analysis.get("skin_condition"),
            "detected_issues": analysis.get("detected_issues", []),
            "suggested_params": analysis.get("suggested_params"),
            "confidence": analysis.get("confidence"),
        }

    try:
        await store.aput(namespace, "record", record)
        logger.info(f"Saved session record: {session_id}")
    except Exception as e:
        logger.error(f"Failed to save session {session_id}: {e}")


async def get_session_history(
    store,
    user_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """
    Retrieve paginated session history for a user.

    Args:
        store: LangGraph Store instance.
        user_id: User identifier.
        limit: Maximum number of sessions to return.
        offset: Number of sessions to skip.

    Returns:
        List of session records, most recent first.
    """
    namespace = ("users", user_id, "sessions")

    sessions = []
    try:
        # asearch returns a list directly (not async iterator)
        items = await store.asearch(namespace, limit=limit, offset=offset)
        for item in items:
            if hasattr(item, 'value') and item.value:
                sessions.append(item.value)
            elif isinstance(item, dict):
                sessions.append(item)
    except Exception as e:
        logger.error(f"Failed to retrieve session history for {user_id}: {e}")

    # Sort by created_at descending
    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return sessions
