"""
Node 4: present_plan

Human-in-the-Loop (HITL) interrupt point 1.
Pauses the workflow and returns the analysis results to the user
for confirmation or adjustment.
"""

import logging
from typing import Any, Optional
from langgraph.types import RunnableConfig
from langgraph.types import interrupt

from agent_face.langgraph_brain.state import BeautifyWorkflowState

logger = logging.getLogger(__name__)


async def present_plan(
    state: BeautifyWorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """
    Present the beautification analysis plan to the user.

    This node calls LangGraph's `interrupt()` to pause execution.
    The user can:
    - Confirm the suggested parameters as-is
    - Adjust specific parameters
    - Request a re-analysis (route back to analyze_image)

    The user's decision is received via `Command(resume=...)` from the API.
    """
    session_id = state.get("session_id", "unknown")
    analysis = state.get("analysis_result", {})
    user_adjustments = state.get("user_adjustments")

    logger.info(f"present_plan: presenting plan for session {session_id}")

    # Build the presentation payload for the HITL interrupt
    plan_payload = {
        "skin_tone": analysis.get("skin_tone", "unknown"),
        "skin_condition": analysis.get("skin_condition", "unknown"),
        "detected_issues": analysis.get("detected_issues", []),
        "suggested_params": analysis.get("suggested_params", {}),
        "reasoning": analysis.get("reasoning", ""),
        "confidence": analysis.get("confidence", 0.0),
    }

    # If the user already provided adjustments (e.g., on retry),
    # skip the interrupt and proceed.
    if user_adjustments:
        logger.info(f"present_plan: user already provided adjustments, merging")
        merged_params = _merge_params(
            analysis.get("suggested_params", {}),
            user_adjustments,
        )
        return {
            "workflow_stage": "plan_confirmed",
            "user_adjustments": user_adjustments,
            "final_params": merged_params,
        }

    # HITL interrupt — workflow pauses here
    user_decision = interrupt(plan_payload)

    # Process the user's decision (received via Command(resume=...))
    action = user_decision.get("action", "confirm")
    adjustments = user_decision.get("adjustments")

    if action == "adjust" and adjustments:
        # User adjusted the parameters
        merged_params = _merge_params(
            analysis.get("suggested_params", {}),
            adjustments,
        )
        logger.info(f"present_plan: user adjusted params for session {session_id}")
        return {
            "workflow_stage": "plan_confirmed",
            "user_adjustments": adjustments,
            "final_params": merged_params,
        }

    # User confirmed without changes
    logger.info(f"present_plan: user confirmed plan for session {session_id}")
    return {
        "workflow_stage": "plan_confirmed",
        "final_params": analysis.get("suggested_params", {}),
    }


def _merge_params(
    suggested: dict[str, float],
    adjustments: dict[str, float],
) -> dict[str, float]:
    """
    Merge user adjustments with suggested parameters.

    User adjustments take precedence for any non-zero values.
    """
    merged = dict(suggested)
    for key, value in adjustments.items():
        if value is not None and value > 0:
            merged[key] = value
    return merged
