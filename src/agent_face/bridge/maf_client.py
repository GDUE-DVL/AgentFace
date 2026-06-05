"""
MAFBridgeClient — the single communication bridge between LangGraph and MAF.

This is the ONLY point of contact between the two frameworks.
LangGraph nodes call methods on this client; the client translates
to MAF's internal message format, invokes the MAF workflow,
and translates the response back.

Design principle: Thin, well-typed, testable. If MAF needs to run
in a separate process, only this file changes — the LangGraph nodes
are unaffected.
"""

import logging
from typing import Optional

from agent_face.langgraph_brain.state import BeautifyParams, AnalysisResult
from agent_face.bridge.types import (
    AnalysisRequest,
    AnalysisResponse,
    BeautificationRequest,
    BeautificationResponse,
)
from agent_face.maf_body.orchestrator import MAFOrchestrator, MAFTaskResult

logger = logging.getLogger(__name__)


class BridgeError(Exception):
    """Error raised when the MAF bridge encounters a failure."""

    def __init__(self, message: str, task_result: Optional[MAFTaskResult] = None):
        super().__init__(message)
        self.task_result = task_result


class MAFBridgeClient:
    """
    Thin adapter bridging LangGraph nodes to the MAF orchestrator.

    LangGraph nodes should never call MAF directly — they always
    go through this client. This ensures:
    - Clean separation of concerns
    - Testability (mock this client in LangGraph tests)
    - Future-proofing (swap to A2A-over-HTTP by changing only this class)
    """

    def __init__(self, orchestrator: MAFOrchestrator):
        self._orchestrator = orchestrator

    # ── Public API ────────────────────────────────────────────

    async def analyze_image(
        self,
        request: AnalysisRequest,
        user_id: str = "anonymous",
        session_id: str = "unknown",
    ) -> AnalysisResponse:
        """
        Analyze a facial image via the multimodal model.

        Called by LangGraph's `analyze_image` node.

        Args:
            request: AnalysisRequest with image, prompt, and preferences.
            user_id: User identifier for tracing.
            session_id: Session identifier for tracing.

        Returns:
            AnalysisResponse with structured analysis result.

        Raises:
            BridgeError: If the MAF task fails.
        """
        logger.info(
            "Bridge: delegating analysis",
            extra={"session_id": session_id, "user_id": user_id},
        )

        task_result = await self._orchestrator.delegate_analysis(
            image_b64=request.image_b64,
            prompt=request.user_prompt,
            preferences=request.user_preferences,
            user_id=user_id,
            session_id=session_id,
            session_count=request.session_count,
            avg_satisfaction=request.avg_satisfaction,
        )

        if not task_result.success:
            raise BridgeError(
                f"Analysis failed: {task_result.error}",
                task_result=task_result,
            )

        return AnalysisResponse(
            result=task_result.result,
            latency_ms=task_result.latency_ms,
            safety_checks=task_result.safety_checks,
        )

    async def apply_beautification(
        self,
        request: BeautificationRequest,
        user_id: str = "anonymous",
        session_id: str = "unknown",
    ) -> BeautificationResponse:
        """
        Apply beautification to a facial image.

        Called by LangGraph's `apply_beautification` node.

        Args:
            request: BeautificationRequest with image and params.
            user_id: User identifier for tracing.
            session_id: Session identifier for tracing.

        Returns:
            BeautificationResponse with base64-encoded result image.

        Raises:
            BridgeError: If the MAF task fails.
        """
        logger.info(
            "Bridge: delegating beautification",
            extra={"session_id": session_id, "user_id": user_id},
        )

        task_result = await self._orchestrator.delegate_beautification(
            image_b64=request.image_b64,
            params=request.params,
            user_id=user_id,
            session_id=session_id,
        )

        if not task_result.success:
            raise BridgeError(
                f"Beautification failed: {task_result.error}",
                task_result=task_result,
            )

        return BeautificationResponse(
            image_b64=task_result.result,
            latency_ms=task_result.latency_ms,
            safety_checks=task_result.safety_checks,
        )

    async def health_check(self) -> dict:
        """Check that MAF orchestrator and all agents are reachable."""
        return await self._orchestrator.health_check()
