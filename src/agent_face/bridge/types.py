"""
Bridge-specific type definitions.

These types define the contract between LangGraph nodes and the
MAF Bridge client. They are intentionally simple dataclasses
to minimize coupling between the two frameworks.
"""

from dataclasses import dataclass, field
from typing import Optional
from agent_face.langgraph_brain.state import BeautifyParams, AnalysisResult


@dataclass
class AnalysisRequest:
    """Request to analyze an image via the multimodal model."""

    image_b64: str
    user_prompt: Optional[str] = None
    user_preferences: Optional[BeautifyParams] = None
    session_count: int = 0
    avg_satisfaction: float = 0.0


@dataclass
class AnalysisResponse:
    """Response from the multimodal analysis."""

    result: AnalysisResult
    latency_ms: float
    model_version: str = "unknown"
    safety_checks: dict = field(default_factory=dict)


@dataclass
class BeautificationRequest:
    """Request to beautify an image via the beautification model."""

    image_b64: str
    params: BeautifyParams
    src_prompt: str = ""
    target_prompt: str = ""
    # Normalized defect boxes from the visual audit.  These are optional; an
    # empty list preserves the original full-face HEdit path.
    edit_regions: list[dict] = field(default_factory=list)
    # Optional per-attempt seed.  Reruns use this to guarantee a new result;
    # ordinary runs leave it unset and use the model service default.
    seed: Optional[int] = None


@dataclass
class BeautificationResponse:
    """Response from the beautification model."""

    image_b64: str  # base64-encoded result
    latency_ms: float
    model_version: str = "unknown"
    safety_checks: dict = field(default_factory=dict)
