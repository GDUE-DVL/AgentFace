"""
LangGraph state machine state definition.

The BeautifyWorkflowState is the single source of truth that flows
through all LangGraph nodes. It carries input data, intermediate
analysis results, user feedback, and workflow control metadata.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from datetime import datetime


class BeautifyParams(TypedDict, total=False):
    """Beautification parameter set (0-5 scale for each dimension)."""

    skin_smoothing: float  # 磨皮
    whitening: float  # 美白
    eye_enlargement: float  # 大眼
    face_slimming: float  # 瘦脸
    blush: float  # 腮红
    lip_color_adjustment: float  # 唇色调整
    blemish_removal: float  # 祛痘/祛斑
    nose_reshaping: float  # 鼻子塑形
    eyebrow_adjustment: float  # 眉毛调整


DEFAULT_BEAUTIFY_PARAMS: BeautifyParams = {
    "skin_smoothing": 2.0,
    "whitening": 1.0,
    "eye_enlargement": 1.5,
    "face_slimming": 1.0,
    "blush": 1.0,
    "lip_color_adjustment": 1.0,
    "blemish_removal": 3.0,
    "nose_reshaping": 0.5,
    "eyebrow_adjustment": 1.0,
}

BEAUTIFY_PARAM_LABELS: dict[str, str] = {
    "skin_smoothing": "磨皮",
    "whitening": "美白",
    "eye_enlargement": "大眼",
    "face_slimming": "瘦脸",
    "blush": "腮红",
    "lip_color_adjustment": "唇色调整",
    "blemish_removal": "祛痘祛斑",
    "nose_reshaping": "鼻子塑形",
    "eyebrow_adjustment": "眉毛调整",
}


class AnalysisResult(TypedDict):
    """Output from the multimodal model's image analysis."""

    skin_tone: str  # e.g., "白皙", "自然", "小麦"
    skin_condition: str  # e.g., "好", "有痘痘", "有斑点", "毛孔粗大"
    detected_features: list[str]  # e.g., ["双眼皮", "圆脸", "高鼻梁"]
    detected_issues: list[str]  # e.g., ["黑眼圈", "痘痘", "色斑"]
    lighting: str  # e.g., "自然光", "暖光", "冷光", "暗光", "强光"
    suggested_params: BeautifyParams
    reasoning: str  # Human-readable explanation of the suggested params
    confidence: float  # 0.0 - 1.0


class FeedbackData(TypedDict, total=False):
    """User feedback after reviewing the beautified image."""

    satisfaction_score: int  # 1-5
    param_adjustments: Optional[BeautifyParams]
    comments: Optional[str]


class BeautifyWorkflowState(TypedDict, total=False):
    """
    Complete state for a beautification workflow session.

    This flows through all LangGraph nodes and is snapshotted
    at each superstep by the checkpointer.
    """

    # -- session metadata --
    session_id: str
    user_id: str
    created_at: str

    # -- input data --
    input_image_b64: str  # base64-encoded original image
    user_prompt: Optional[str]  # optional natural language instruction

    # -- context (loaded from long-term store) --
    user_preferences: Optional[BeautifyParams]
    style_profile: Optional[dict]

    # -- analysis phase --
    analysis_result: Optional[AnalysisResult]
    user_adjustments: Optional[BeautifyParams]  # HITL overrides

    # -- beautification phase --
    final_params: Optional[BeautifyParams]  # merged params to apply
    beautified_image_b64: Optional[str]  # base64 result

    # -- feedback phase --
    user_feedback: Optional[FeedbackData]

    # -- workflow control --
    workflow_stage: str  # current stage name
    error_message: Optional[str]
    retry_count: int

    # -- conversation messages (managed by add_messages reducer) --
    messages: Annotated[list[BaseMessage], add_messages]
