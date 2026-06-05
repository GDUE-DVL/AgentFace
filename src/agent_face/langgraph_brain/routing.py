"""
LangGraph conditional routing functions.

Each function inspects the state and returns the name of the
next node to execute.
"""

import logging
from agent_face.langgraph_brain.state import BeautifyWorkflowState

logger = logging.getLogger(__name__)


def route_after_input(state: BeautifyWorkflowState) -> str:
    """
    Route after receive_input.

    If validation failed, go to handle_error.
    Otherwise, proceed to retrieve_context.
    """
    if state.get("workflow_stage") == "input_failed":
        logger.warning("route_after_input: input validation failed, routing to handle_error")
        return "handle_error"
    return "retrieve_context"


def route_after_analysis(state: BeautifyWorkflowState) -> str:
    """
    Route after analyze_image.

    If analysis succeeded, proceed to present_plan.
    If analysis failed, route to handle_error.
    """
    if state.get("workflow_stage") == "error":
        logger.warning("route_after_analysis: analysis failed, routing to handle_error")
        return "handle_error"
    return "present_plan"


def route_after_presentation(state: BeautifyWorkflowState) -> str:
    """
    Route after present_plan (HITL interrupt).

    Normally proceeds to apply_beautification.
    But if the user requested a re-analysis, route back to analyze_image.
    """
    stage = state.get("workflow_stage", "")
    if stage == "plan_confirmed":
        return "apply_beautification"
    # User may have requested re-analysis via HITL
    return "apply_beautification"


def route_after_beautification(state: BeautifyWorkflowState) -> str:
    """
    Route after apply_beautification.

    If beautification succeeded, proceed to collect_feedback.
    If beautification failed, route to handle_error.
    """
    if state.get("workflow_stage") == "error":
        logger.warning("route_after_beautification: beautification failed, routing to handle_error")
        return "handle_error"
    return "collect_feedback"


def route_after_error(state: BeautifyWorkflowState) -> str:
    """
    Route after handle_error.

    If we're retrying, determine the recovery point based on
    where the error occurred.
    If we've exhausted retries, go to finalize (fail gracefully).
    """
    stage = state.get("workflow_stage", "")
    retry_count = state.get("retry_count", 0)

    if stage == "retrying" and retry_count <= 3:
        # Determine recovery point: if we have analysis results,
        # we failed during beautification; otherwise during analysis.
        if state.get("analysis_result"):
            logger.info("route_after_error: retrying from apply_beautification")
            return "apply_beautification"
        else:
            logger.info("route_after_error: retrying from analyze_image")
            return "analyze_image"

    # Give up — go to finalize to return best-effort result
    logger.warning("route_after_error: exhausted retries, routing to finalize")
    return "finalize"
