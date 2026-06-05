"""
MAF Orchestrator — central coordinator for the Microsoft Agent Framework body.

The orchestrator manages the DAG workflow:
    1. SafetyGuardAgent (input check)
    2. SpecialistAgent (Multimodal or Beautification, depending on intent)
    3. SafetyGuardAgent (output check)

It uses A2A-style task delegation to communicate with sub-agents
and applies the middleware pipeline to every agent execution.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Any

from agent_face.langgraph_brain.state import BeautifyParams, AnalysisResult
from agent_face.maf_body.agents.safety_guard_agent import SafetyGuardAgent, SafetyResult
from agent_face.maf_body.agents.multimodal_agent import MultimodalAgent
from agent_face.maf_body.agents.beautification_agent import BeautificationAgent
from agent_face.maf_body.middleware.pipeline import MiddlewarePipeline, MiddlewareContext

logger = logging.getLogger(__name__)


@dataclass
class MAFTaskResult:
    """Result of a task executed by the MAF orchestrator."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    safety_checks: dict = None
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.safety_checks is None:
            self.safety_checks = {}


class MAFOrchestrator:
    """
    MAF Orchestrator — uses a DAG pattern to coordinate agents.

    This is the single entry point into the MAF "body" from the Bridge layer.
    It:
    1. Receives tasks (analyze or beautify)
    2. Runs input safety guards
    3. Delegates to specialist agents via A2A-style messages
    4. Runs output safety guards
    5. Returns results through the middleware pipeline

    The DAG topology is:
        Orchestrator
          ├── SafetyGuardAgent (input)
          │     └── SpecialistAgent (multimodal or beautification)
          └── SafetyGuardAgent (output)
    """

    def __init__(self):
        self._safety_guard = SafetyGuardAgent()
        self._multimodal_agent = MultimodalAgent()
        self._beautification_agent = BeautificationAgent()
        self._middleware = MiddlewarePipeline()
        self._started = False

    async def start(self):
        """Initialize and start all agents."""
        self._started = True
        logger.info("MAF Orchestrator started")

    async def stop(self):
        """Shutdown all agents."""
        self._started = False
        logger.info("MAF Orchestrator stopped")

    # ── Public API (called by Bridge) ─────────────────────────

    async def delegate_analysis(
        self,
        image_b64: str,
        prompt: Optional[str] = None,
        preferences: Optional[BeautifyParams] = None,
        user_id: str = "anonymous",
        session_id: str = "unknown",
        session_count: int = 0,
        avg_satisfaction: float = 0.0,
    ) -> MAFTaskResult:
        """
        Delegate a facial analysis task.

        Workflow:
        1. SafetyGuardAgent validates the input image
        2. MultimodalAgent analyzes the face
        3. Returns AnalysisResult

        Args:
            image_b64: Base64-encoded facial image.
            prompt: Optional natural language instruction.
            preferences: Optional user aesthetic preferences.
            user_id: User identifier for logging/compliance.
            session_id: Session identifier for tracing.

        Returns:
            MAFTaskResult with AnalysisResult on success.
        """
        start_time = time.monotonic()

        try:
            # Step 1: Input safety check
            safety_result = await self._safety_guard.validate_input(image_b64)
            if not safety_result.passed:
                return MAFTaskResult(
                    success=False,
                    error=f"Safety check failed: {'; '.join(safety_result.violations)}",
                    safety_checks={"input": safety_result},
                    latency_ms=(time.monotonic() - start_time) * 1000,
                )

            # Use downsampled image if available
            working_image = safety_result.downsampled_image_b64 or image_b64

            # Step 2: Run multimodal analysis through middleware pipeline
            input_data = {
                "image_b64": working_image,
                "prompt": prompt,
                "preferences": preferences,
            }
            context = MiddlewareContext(
                user_id=user_id,
                session_id=session_id,
                task_type="analyze",
                start_time=start_time,
            )

            async def run_analysis(data: dict) -> dict:
                result = await self._multimodal_agent.analyze(
                    image_b64=data["image_b64"],
                    prompt=data.get("prompt"),
                    preferences=data.get("preferences"),
                    session_count=session_count,
                    avg_satisfaction=avg_satisfaction,
                )
                return {"analysis_result": result}

            output = await self._middleware.execute(input_data, context, run_analysis)

            latency = (time.monotonic() - start_time) * 1000
            return MAFTaskResult(
                success=True,
                result=output["analysis_result"],
                safety_checks={
                    "input": safety_result,
                    "warnings": safety_result.warnings,
                },
                latency_ms=latency,
            )

        except Exception as e:
            logger.exception("Analysis delegation failed")
            return MAFTaskResult(
                success=False,
                error=str(e),
                latency_ms=(time.monotonic() - start_time) * 1000,
            )

    async def delegate_beautification(
        self,
        image_b64: str,
        params: BeautifyParams,
        user_id: str = "anonymous",
        session_id: str = "unknown",
    ) -> MAFTaskResult:
        """
        Delegate a beautification task.

        Workflow:
        1. SafetyGuardAgent validates the input image
        2. BeautificationAgent applies the parameters
        3. SafetyGuardAgent validates the output (over-smoothing, etc.)
        4. Returns the beautified image

        Args:
            image_b64: Base64-encoded facial image.
            params: Beautification parameters to apply.
            user_id: User identifier for logging/compliance.
            session_id: Session identifier for tracing.

        Returns:
            MAFTaskResult with base64-encoded beautified image on success.
        """
        start_time = time.monotonic()

        try:
            # Step 1: Input safety check
            safety_result = await self._safety_guard.validate_input(image_b64)
            if not safety_result.passed:
                return MAFTaskResult(
                    success=False,
                    error=f"Safety check failed: {'; '.join(safety_result.violations)}",
                    safety_checks={"input": safety_result},
                    latency_ms=(time.monotonic() - start_time) * 1000,
                )

            working_image = safety_result.downsampled_image_b64 or image_b64

            # Step 2: Run beautification through middleware pipeline
            input_data = {
                "image_b64": working_image,
                "params": params,
            }
            context = MiddlewareContext(
                user_id=user_id,
                session_id=session_id,
                task_type="beautify",
                start_time=start_time,
            )

            async def run_beautification(data: dict) -> dict:
                result = await self._beautification_agent.beautify(
                    image_b64=data["image_b64"],
                    params=data["params"],
                )
                return {"beautified_image_b64": result}

            output = await self._middleware.execute(input_data, context, run_beautification)

            beautified_b64 = output["beautified_image_b64"]

            # Step 3: Output safety check (over-smoothing, etc.)
            output_safety = await self._safety_guard.validate_output(
                original_b64=working_image,
                result_b64=beautified_b64,
            )

            latency = (time.monotonic() - start_time) * 1000
            return MAFTaskResult(
                success=True,
                result=beautified_b64,
                safety_checks={
                    "input": safety_result,
                    "output": output_safety,
                    "warnings": safety_result.warnings + output_safety.warnings,
                },
                latency_ms=latency,
            )

        except Exception as e:
            logger.exception("Beautification delegation failed")
            return MAFTaskResult(
                success=False,
                error=str(e),
                latency_ms=(time.monotonic() - start_time) * 1000,
            )

    async def health_check(self) -> dict:
        """Check health of all agents and tools."""
        return {
            "status": "ok" if self._started else "not_started",
            "safety_guard": await self._safety_guard.health_check(),
            "multimodal_agent": await self._multimodal_agent.health_check(),
            "beautification_agent": await self._beautification_agent.health_check(),
        }
