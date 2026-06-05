"""
User feedback aggregation and trend analysis.

Aggregates feedback across sessions to detect trends:
- Average satisfaction
- Common complaints
- Parameter satisfaction correlation
"""

import logging
from typing import Optional

from agent_face.langgraph_brain.state import BeautifyParams, FeedbackData

logger = logging.getLogger(__name__)

# Keep only the last N satisfaction scores for trend analysis
MAX_TREND_HISTORY = 50


async def record_feedback(
    store,
    user_id: str,
    session_id: str,
    feedback: FeedbackData,
    session_params: Optional[BeautifyParams] = None,
) -> dict:
    """
    Record user feedback and update aggregate statistics.

    Called by the `update_memory` node.

    Args:
        store: LangGraph Store instance.
        user_id: User identifier.
        session_id: Session identifier.
        feedback: User feedback data.
        session_params: The parameters used in this session.

    Returns:
        Updated aggregate feedback dict.
    """
    namespace = ("users", user_id, "feedback")
    satisfaction = feedback.get("satisfaction_score", 0)

    # Load existing aggregate
    aggregate = {
        "total_sessions": 0,
        "avg_satisfaction": 0.0,
        "satisfaction_trend": [],
        "common_complaints": [],
        "param_satisfaction_correlation": {},
    }

    try:
        item = await store.aget(namespace, "aggregate")
        if item and item.value:
            aggregate.update(item.value)
    except Exception:
        pass

    # Update statistics
    total = aggregate["total_sessions"] + 1
    old_avg = aggregate["avg_satisfaction"]
    new_avg = round((old_avg * (total - 1) + satisfaction) / total, 2)

    # Update trend (keep last N)
    trend = aggregate.get("satisfaction_trend", [])
    trend.append(satisfaction)
    if len(trend) > MAX_TREND_HISTORY:
        trend = trend[-MAX_TREND_HISTORY:]

    # Update complaints
    comments = feedback.get("comments", "")
    complaints = list(aggregate.get("common_complaints", []))
    if comments and satisfaction <= 3:
        # Simple keyword-based complaint detection
        complaint_keywords = {
            "too_smooth": ["太光滑", "太磨", "不真实", "假"],
            "too_white": ["太白", "假白", "太亮"],
            "too_dark": ["太黑", "太暗"],
            "unnatural": ["不自然", "怪", "奇怪"],
            "eye_too_big": ["眼睛太大", "大眼过度"],
            "face_too_thin": ["脸太瘦", "太尖"],
        }
        for issue, keywords in complaint_keywords.items():
            if any(kw in comments.lower() for kw in keywords):
                existing = next((c for c in complaints if c["issue"] == issue), None)
                if existing:
                    existing["count"] += 1
                else:
                    complaints.append({"issue": issue, "count": 1})

    # Update parameter satisfaction correlation (coarse)
    correlation = dict(aggregate.get("param_satisfaction_correlation", {}))
    if session_params:
        for key in session_params:
            if key not in correlation:
                correlation[key] = {"high_satisfaction_count": 0, "low_satisfaction_count": 0,
                                     "high_total_sessions": 0, "low_total_sessions": 0}
            if session_params[key] >= 3.0:
                correlation[key]["high_total_sessions"] += 1
                if satisfaction >= 3:
                    correlation[key]["high_satisfaction_count"] += 1
            else:
                correlation[key]["low_total_sessions"] += 1
                if satisfaction >= 3:
                    correlation[key]["low_satisfaction_count"] += 1

    # Save aggregate
    updated_aggregate = {
        "total_sessions": total,
        "avg_satisfaction": new_avg,
        "satisfaction_trend": trend,
        "common_complaints": complaints,
        "param_satisfaction_correlation": correlation,
    }

    try:
        await store.aput(namespace, "aggregate", updated_aggregate)
        logger.info(f"Recorded feedback for session {session_id}: score={satisfaction}")
    except Exception as e:
        logger.error(f"Failed to store feedback for {session_id}: {e}")

    return updated_aggregate


async def get_user_feedback_summary(store, user_id: str) -> dict:
    """
    Get the aggregated feedback summary for a user.

    Args:
        store: LangGraph Store instance.
        user_id: User identifier.

    Returns:
        Aggregated feedback dict.
    """
    try:
        item = await store.aget(("users", user_id, "feedback"), "aggregate")
        if item and item.value:
            return item.value
    except Exception as e:
        logger.error(f"Failed to load feedback summary for {user_id}: {e}")

    return {
        "total_sessions": 0,
        "avg_satisfaction": 0.0,
        "satisfaction_trend": [],
        "common_complaints": [],
        "param_satisfaction_correlation": {},
    }
