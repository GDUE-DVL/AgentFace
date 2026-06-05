"""
Pytest fixtures for AgentFace tests.

Provides mock fixtures for all major components:
- Mock MAF orchestrator (for testing LangGraph in isolation)
- Mock bridge client
- In-memory graph with checkpointer and store
- Sample state fixtures
"""

import base64
import io
import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image

from agent_face.langgraph_brain.graph import build_graph
from agent_face.langgraph_brain.state import (
    BeautifyWorkflowState,
    BeautifyParams,
    AnalysisResult,
    DEFAULT_BEAUTIFY_PARAMS,
)
from agent_face.bridge.maf_client import MAFBridgeClient
from agent_face.bridge.types import AnalysisResponse, BeautificationResponse


# ── Sample Data ────────────────────────────────────────────────


def _create_test_image_b64() -> str:
    """Create a small test image as base64."""
    img = Image.new("RGB", (100, 100), color="pink")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def sample_image_b64() -> str:
    return _create_test_image_b64()


@pytest.fixture
def sample_state() -> BeautifyWorkflowState:
    return {
        "session_id": "test-session-001",
        "user_id": "test-user-001",
        "input_image_b64": _create_test_image_b64(),
        "user_prompt": "make me look natural",
        "created_at": "2026-06-03T10:00:00Z",
        "workflow_stage": "start",
        "retry_count": 0,
    }


@pytest.fixture
def sample_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        skin_tone="白皙",
        skin_condition="良好",
        detected_features=["双眼皮", "瓜子脸"],
        detected_issues=["轻微黑眼圈"],
        suggested_params=BeautifyParams(
            skin_smoothing=2.0,
            whitening=1.5,
            eye_enlargement=1.0,
            face_slimming=0.5,
            blush=1.0,
            lip_color_adjustment=1.0,
            blemish_removal=2.0,
            nose_reshaping=0.5,
            eyebrow_adjustment=1.0,
        ),
        reasoning="自然肤色，轻微黑眼圈建议轻度遮瑕",
        confidence=0.88,
    )


# ── Mock MAF Components ────────────────────────────────────────


class MockMAFOrchestrator:
    """Mock MAF orchestrator for isolated LangGraph testing."""

    def __init__(self, analysis_result=None, beautified_image_b64=None):
        self.analysis_result = analysis_result
        self.beautified_image_b64 = beautified_image_b64 or _create_test_image_b64()
        self._started = False
        self.analysis_calls = []
        self.beautification_calls = []

    async def start(self):
        self._started = True

    async def stop(self):
        self._started = False

    async def delegate_analysis(self, image_b64, prompt=None, preferences=None,
                                user_id="anonymous", session_id="unknown"):
        self.analysis_calls.append({
            "image_b64": image_b64,
            "prompt": prompt,
            "preferences": preferences,
            "user_id": user_id,
            "session_id": session_id,
        })
        from agent_face.maf_body.orchestrator import MAFTaskResult

        if self.analysis_result is None:
            return MAFTaskResult(
                success=True,
                result=AnalysisResult(
                    skin_tone="自然",
                    skin_condition="好",
                    detected_features=[],
                    detected_issues=[],
                    suggested_params=DEFAULT_BEAUTIFY_PARAMS,
                    reasoning="默认分析结果",
                    confidence=0.5,
                ),
                latency_ms=100.0,
            )

        return MAFTaskResult(
            success=True,
            result=self.analysis_result,
            latency_ms=100.0,
        )

    async def delegate_beautification(self, image_b64, params, user_id="anonymous",
                                       session_id="unknown"):
        self.beautification_calls.append({
            "image_b64": image_b64,
            "params": params,
            "user_id": user_id,
            "session_id": session_id,
        })
        from agent_face.maf_body.orchestrator import MAFTaskResult

        return MAFTaskResult(
            success=True,
            result=self.beautified_image_b64,
            latency_ms=200.0,
            safety_checks={"warnings": []},
        )

    async def health_check(self):
        return {"status": "ok"}


@pytest.fixture
def mock_orchestrator(sample_analysis_result):
    return MockMAFOrchestrator(analysis_result=sample_analysis_result)


@pytest.fixture
def mock_bridge(mock_orchestrator):
    return MAFBridgeClient(orchestrator=mock_orchestrator)


# ── Graph Fixtures ─────────────────────────────────────────────


@pytest.fixture
def test_graph():
    """Build a graph with in-memory checkpointer and store."""
    return build_graph(db_path=None)  # In-memory for testing


@pytest.fixture
def graph_config(test_graph, mock_bridge):
    """Create a graph config with bridge, store, and thread_id."""
    return {
        "configurable": {
            "thread_id": "test-thread-001",
            "bridge": mock_bridge,
            "store": test_graph.store,
        }
    }
