"""
Node 2: retrieve_context

Loads user preferences, style profile, and recent session history
from the LangGraph Store to provide context for the analysis.
"""

import logging
from typing import Any, Optional
from langgraph.types import RunnableConfig

from agent_face.langgraph_brain.state import BeautifyWorkflowState
from agent_face.langgraph_brain.memory.preferences import get_user_preferences

logger = logging.getLogger(__name__)


async def retrieve_context(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Load user context from long-term memory.

    Retrieves:
    - User beautification preferences
    - Style profile (if available)
    - Recent session history for trend context

    The Store is accessed via config["configurable"]["store"].
    """
    store = (config.get("configurable", {}) if config else {}).get("store")
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "unknown")

    logger.info(f"retrieve_context: loading context for user {user_id}")

    if store is None:
        logger.warning("No Store available in config, using defaults")
        return {
            "workflow_stage": "context_loaded",
            "user_preferences": None,
            "style_profile": None,
        }

    # Load preferences
    preferences = await get_user_preferences(store, user_id)

    # Load style profile
    style_profile = None
    try:
        item = await store.aget(("users", user_id, "style_profile"), "current")
        if item and item.value:
            style_profile = item.value
    except Exception as e:
        logger.warning(f"Failed to load style profile for {user_id}: {e}")

    logger.info(
        f"retrieve_context: loaded context for {user_id} "
        f"(prefs={bool(preferences)}, style={bool(style_profile)})"
    )

    return {
        "workflow_stage": "context_loaded",
        "user_preferences": preferences,
        "style_profile": style_profile,
    }
