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


@dataclass
class BeautificationResponse:
    """Response from the beautification model."""

    image_b64: str  # base64-encoded result
    latency_ms: float
    model_version: str = "unknown"
    safety_checks: dict = field(default_factory=dict)
