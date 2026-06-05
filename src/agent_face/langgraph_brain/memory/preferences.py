"""
User preference memory with bidirectional learning.

- 评分 4-5（满意）：正向学习，偏好向本次参数靠拢
- 评分 3   （一般）：轻微正向学习
- 评分 1-2（不满意）：反向反思，偏好远离本次参数

Score → direction & strength:
  5 → +alpha       (强力靠拢)
  4 → +alpha*0.7   (靠拢)
  3 → +alpha*0.3   (轻微靠拢)
  2 → -alpha*0.7   (远离)
  1 → -alpha       (强力远离)
"""

import logging
from typing import Optional
from agent_face.langgraph_brain.state import BeautifyParams, DEFAULT_BEAUTIFY_PARAMS
from agent_face.config import settings

logger = logging.getLogger(__name__)

# Base learning rate
ALPHA = settings.preference_learning_rate   # 0.4
CLAMP_MIN = 0.0
CLAMP_MAX = 5.0


def _score_to_alpha(satisfaction_score: int) -> float:
    """Convert satisfaction score (1-5) to a signed learning rate.

    Positive alpha → preference moves toward session params (like)
    Negative alpha → preference moves away from session params (dislike)

    1 → -0.40  强力反思（远离）
    2 → -0.28  反思（远离）
    3 → +0.12  轻微学习（靠拢）
    4 → +0.28  学习（靠拢）
    5 → +0.40  强力学习（靠拢）
    """
    strength_map = {
        1: -1.0,
        2: -0.7,
        3: +0.3,
        4: +0.7,
        5: +1.0,
    }
    return ALPHA * strength_map.get(satisfaction_score, 0)


async def get_user_preferences(
    store, user_id: str
) -> BeautifyParams:
    """Load user's current beautification preferences from the Store."""
    namespace = ("users", user_id, "preferences")
    try:
        item = await store.aget(namespace, "current")
        if item and item.value:
            prefs = dict(DEFAULT_BEAUTIFY_PARAMS)
            prefs.update(item.value)
            return BeautifyParams(**prefs)
    except Exception as e:
        logger.warning(f"Failed to load preferences for {user_id}: {e}")

    return BeautifyParams(**DEFAULT_BEAUTIFY_PARAMS)


async def update_user_preferences(
    store,
    user_id: str,
    session_params: BeautifyParams,
    satisfaction_score: int,
) -> BeautifyParams:
    """
    Bidirectional preference learning from user feedback.

    - positive score → move TOWARD session params (用户喜欢)
    - negative score → move AWAY from session params (用户不喜欢，反思)

    Formula:
      signed_alpha = alpha * direction_factor  (direction factor from score)
      new = stored + signed_alpha * (session - stored)
      clamped to [0.0, 5.0]

    When signed_alpha > 0: new moves toward session (like)
    When signed_alpha < 0: new moves away from session (dislike → reflect)
    """
    signed_alpha = _score_to_alpha(satisfaction_score)
    direction = "靠拢" if signed_alpha > 0 else "远离"

    logger.info(
        f"Preference update for {user_id}: "
        f"score={satisfaction_score}, alpha={signed_alpha:+.2f} ({direction})"
    )

    namespace = ("users", user_id, "preferences")
    stored = await get_user_preferences(store, user_id)

    updated = {}
    for key in DEFAULT_BEAUTIFY_PARAMS:
        stored_val = stored.get(key, DEFAULT_BEAUTIFY_PARAMS[key])
        session_val = session_params.get(key, DEFAULT_BEAUTIFY_PARAMS[key])

        # new = stored + signed_alpha * (session - stored)
        new_val = stored_val + signed_alpha * (session_val - stored_val)

        # Clamp to valid range
        updated[key] = round(max(CLAMP_MIN, min(CLAMP_MAX, new_val)), 2)

    try:
        await store.aput(namespace, "current", updated)
        logger.info(
            f"Prefs updated for {user_id} (score={satisfaction_score}, "
            f"alpha={signed_alpha:+.2f}): "
            f"{ {k: f'{updated[k]:.1f}' for k in list(updated)[:4]} }..."
        )
    except Exception as e:
        logger.error(f"Failed to store preferences for {user_id}: {e}")

    return BeautifyParams(**updated)
