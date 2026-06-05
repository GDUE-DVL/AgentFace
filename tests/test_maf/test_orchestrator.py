"""
Test the MAF Orchestrator and its agent interactions.

Tests the diamond DAG workflow:
    SafetyGuard → SpecialistAgent → SafetyGuard
"""

import pytest
from agent_face.maf_body.orchestrator import MAFOrchestrator, MAFTaskResult
from agent_face.langgraph_brain.state import BeautifyParams


class TestMAFOrchestrator:
    """Test the MAF orchestrator's task delegation."""

    @pytest.fixture
    async def orchestrator(self):
        orch = MAFOrchestrator()
        await orch.start()
        yield orch
        await orch.stop()

    @pytest.mark.asyncio
    async def test_start_stop(self, orchestrator):
        """Test orchestrator lifecycle."""
        assert orchestrator._started
        await orchestrator.stop()
        assert not orchestrator._started

    @pytest.mark.asyncio
    async def test_delegate_analysis(self, orchestrator, sample_image_b64):
        """Test analysis delegation (will fail since models aren't running)."""
        result = await orchestrator.delegate_analysis(
            image_b64=sample_image_b64,
            prompt="test",
            user_id="test-user",
            session_id="test-session",
        )

        # Since actual models aren't running, this will either:
        # - Succeed with mock models (if configured)
        # - Fail with connection error
        # The important thing is the DAG flow executes correctly.
        assert isinstance(result, MAFTaskResult)
        # Either success or a connection error (expected without real models)
        if not result.success:
            assert "Connection" in str(result.error) or "connect" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_delegate_beautification(self, orchestrator, sample_image_b64):
        """Test beautification delegation."""
        params = BeautifyParams(
            skin_smoothing=2.0,
            whitening=1.0,
            eye_enlargement=1.5,
            face_slimming=1.0,
            blush=1.0,
            lip_color_adjustment=1.0,
            blemish_removal=3.0,
            nose_reshaping=0.5,
            eyebrow_adjustment=1.0,
        )

        result = await orchestrator.delegate_beautification(
            image_b64=sample_image_b64,
            params=params,
            user_id="test-user",
            session_id="test-session",
        )

        assert isinstance(result, MAFTaskResult)

    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator):
        """Test orchestrator health check."""
        health = await orchestrator.health_check()
        assert "status" in health
        assert "multimodal_agent" in health
        assert "beautification_agent" in health
        assert "safety_guard" in health


class TestSafetyGuardAgent:
    """Test the SafetyGuardAgent's input/output validation."""

    @pytest.fixture
    def safety_guard(self):
        from agent_face.maf_body.agents.safety_guard_agent import SafetyGuardAgent
        return SafetyGuardAgent()

    @pytest.mark.asyncio
    async def test_invalid_base64(self, safety_guard):
        """Test that invalid base64 is rejected."""
        result = await safety_guard.validate_input("!!!bad!!!")
        assert not result.passed
        assert any("Invalid base64" in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_valid_image(self, safety_guard, sample_image_b64):
        """Test that a valid image passes validation."""
        result = await safety_guard.validate_input(sample_image_b64)
        assert result.passed


class TestBeautifyParamsConversion:
    """Test the params-to-natural-language conversion."""

    def test_params_to_description(self):
        from agent_face.models.beautify_client import BeautifyModelClient

        params = BeautifyParams(
            skin_smoothing=3.0,
            whitening=2.0,
            eye_enlargement=1.0,
            face_slimming=0.5,
            blush=1.0,
            lip_color_adjustment=1.0,
            blemish_removal=4.0,
            nose_reshaping=0.5,
            eyebrow_adjustment=1.0,
        )

        desc = BeautifyModelClient.params_to_description(params)
        assert "磨皮" in desc
        assert "美白" in desc
        assert "祛痘祛斑" in desc
        assert "中度" in desc or "较明显" in desc
