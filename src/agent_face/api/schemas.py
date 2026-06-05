"""
Pydantic request/response models for the FastAPI layer.

These are separate from the LangGraph TypedDicts — they handle
HTTP serialization, validation, and OpenAPI documentation.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Beautify Params ──────────────────────────────────────────────


class BeautifyParamsModel(BaseModel):
    """Beautification parameters (0.0-5.0 scale)."""

    skin_smoothing: float = Field(default=2.0, ge=0.0, le=5.0, description="磨皮程度")
    whitening: float = Field(default=1.0, ge=0.0, le=5.0, description="美白程度")
    eye_enlargement: float = Field(default=1.5, ge=0.0, le=5.0, description="大眼程度")
    face_slimming: float = Field(default=1.0, ge=0.0, le=5.0, description="瘦脸程度")
    blush: float = Field(default=1.0, ge=0.0, le=5.0, description="腮红程度")
    lip_color_adjustment: float = Field(default=1.0, ge=0.0, le=5.0, description="唇色调整")
    blemish_removal: float = Field(default=3.0, ge=0.0, le=5.0, description="祛痘祛斑")
    nose_reshaping: float = Field(default=0.5, ge=0.0, le=5.0, description="鼻子塑形")
    eyebrow_adjustment: float = Field(default=1.0, ge=0.0, le=5.0, description="眉毛调整")


# ── Analysis Result ──────────────────────────────────────────────


class AnalysisResultModel(BaseModel):
    """Output from the multimodal model's image analysis."""

    skin_tone: str = Field(description="肤色分析")
    skin_condition: str = Field(description="皮肤状况")
    lighting: str = Field(default="—", description="光线状况")
    detected_features: list[str] = Field(default_factory=list, description="检测到的面部特征")
    detected_issues: list[str] = Field(default_factory=list, description="检测到的皮肤问题")
    suggested_params: BeautifyParamsModel = Field(description="建议的美颜参数")
    reasoning: str = Field(description="分析理由")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")


# ── Request Models ───────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    """Request to create a new beautification session."""

    image: str = Field(description="Base64 编码的原始图片")
    user_prompt: Optional[str] = Field(default=None, description="用户自然语言指令（可选）")
    user_id: str = Field(min_length=1, description="用户 ID")


class ConfirmPlanRequest(BaseModel):
    """Request to confirm or adjust the beautification plan."""

    action: str = Field(pattern="^(confirm|adjust)$", description="confirm 确认方案 / adjust 调整方案")
    adjustments: Optional[BeautifyParamsModel] = Field(
        default=None, description="调整后的参数 (仅 action=adjust 时需要)"
    )


class SubmitFeedbackRequest(BaseModel):
    """Request to submit feedback after reviewing the result."""

    satisfaction_score: int = Field(ge=1, le=5, description="满意度评分 1-5")
    param_adjustments: Optional[BeautifyParamsModel] = Field(
        default=None, description="用户希望的手动参数调整"
    )
    comments: Optional[str] = Field(default=None, description="文字反馈")


# ── Response Models ──────────────────────────────────────────────


class SessionResponse(BaseModel):
    """Response for session status queries."""

    session_id: str
    user_id: str
    workflow_stage: str
    status: str  # "analyzing" | "awaiting_confirmation" | "beautifying" | "awaiting_feedback" | "completed" | "error"
    created_at: str

    # Only populated at certain stages
    analysis_result: Optional[AnalysisResultModel] = None
    beautified_image: Optional[str] = None  # base64
    final_params: Optional[BeautifyParamsModel] = None
    error_message: Optional[str] = None


class CreateSessionResponse(BaseModel):
    """Response after creating a session."""

    session_id: str
    status: str


class UserPreferencesResponse(BaseModel):
    """Response for user preferences query."""

    user_id: str
    preferences: BeautifyParamsModel
    style_profile: Optional[dict] = None
    total_sessions: int = 0
    avg_satisfaction: Optional[float] = None


class SessionHistoryItem(BaseModel):
    """A single item in session history."""

    session_id: str
    created_at: str
    workflow_stage: str
    satisfaction_score: Optional[int] = None
    thumbnail_params: Optional[BeautifyParamsModel] = None


class SessionHistoryResponse(BaseModel):
    """Response for session history query."""

    user_id: str
    total: int
    sessions: list[SessionHistoryItem]


class HealthResponse(BaseModel):
    """System health check response."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    langgraph: str
    maf_orchestrator: str
    multimodal_model: dict
    beautification_model: dict


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None
    session_id: Optional[str] = None
