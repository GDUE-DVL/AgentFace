"""
Node 7: update_memory

Writes session data to long-term memory:
- Session record (history)
- Feedback aggregation
- Preference evolution (EMA update if satisfaction >= 3)
"""

import logging
from typing import Any, Optional
from langgraph.types import RunnableConfig

from agent_face.langgraph_brain.state import BeautifyWorkflowState
from agent_face.langgraph_brain.memory.history import save_session_record
from agent_face.langgraph_brain.memory.feedback import record_feedback
from agent_face.langgraph_brain.memory.preferences import update_user_preferences

logger = logging.getLogger(__name__)


async def update_memory(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Persist all session data to long-term memory.

    Writes to three Store namespaces:
    1. Session history: ("users", {id}, "sessions", {sid})
    2. Feedback aggregate: ("users", {id}, "feedback")
    3. Preferences: ("users", {id}, "preferences") — only if satisfied

    Args:
        state: Current workflow state.
        config: LangGraph config containing Store reference.

    Returns:
        State update with workflow_stage = "memory_updated".
    """
    store = (config.get("configurable", {}) if config else {}).get("store")
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "unknown")

    logger.info(f"update_memory: persisting data for session {session_id}")

    if store is None:
        logger.warning("No Store available, skipping memory persistence")
        return {"workflow_stage": "memory_updated"}

    feedback = state.get("user_feedback", {})
    satisfaction = feedback.get("satisfaction_score", 0)
    final_params = state.get("final_params")

    # 1. Save session history
    await save_session_record(store, state)

    # 2. Record feedback and update aggregate statistics
    await record_feedback(
        store=store,
        user_id=user_id,
        session_id=session_id,
        feedback=feedback,
        session_params=final_params,
    )

    # 3. Update preferences — bidirectional learning from all feedback
    if final_params:
        await update_user_preferences(
            store=store,
            user_id=user_id,
            session_params=final_params,
            satisfaction_score=satisfaction,
        )

    logger.info(
        f"update_memory: memory persisted for session {session_id} "
        f"(satisfaction={satisfaction})"
    )

    return {"workflow_stage": "memory_updated"}
