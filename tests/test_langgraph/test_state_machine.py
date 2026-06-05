"""
Test the LangGraph beautification workflow state machine.

Covers:
- Full happy path from START to END
- HITL interrupt at present_plan and collect_feedback
- Error handling and retry logic
- Memory (checkpointer) persistence across sessions
"""

import pytest
from langgraph.types import Command


class TestHappyPath:
    """Test the complete happy path from input to completion."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, test_graph, graph_config, sample_state):
        """
        Test: receive_input → retrieve_context → analyze_image →
              present_plan (interrupt, resume) → apply_beautification →
              collect_feedback (interrupt, resume) → update_memory → finalize
        """
        # Step 1: Run until first HITL interrupt (present_plan)
        try:
            await test_graph.ainvoke(sample_state, graph_config)
        except Exception:
            pass  # Expected: GraphInterrupt

        # Verify we're at present_plan
        state = await test_graph.aget_state(graph_config)
        stage = state.values.get("workflow_stage", "")
        assert stage in ("plan_presented", "analyzed", "plan_confirmed"), \
            f"Expected at plan stage, got: {stage}"

        # Step 2: Confirm the plan and resume
        resume_value = {"action": "confirm"}
        try:
            await test_graph.ainvoke(Command(resume=resume_value), graph_config)
        except Exception:
            pass  # Expected: GraphInterrupt at collect_feedback

        state = await test_graph.aget_state(graph_config)
        stage = state.values.get("workflow_stage", "")
        assert stage in ("beautified",), \
            f"Expected at beautified stage, got: {stage}"
        assert state.values.get("beautified_image_b64") is not None

        # Step 3: Submit feedback and complete
        final_state = await test_graph.ainvoke(
            Command(resume={"satisfaction_score": 4, "comments": "Looks great!"}),
            graph_config,
        )

        assert final_state["workflow_stage"] == "completed"


class TestHITLAdjustments:
    """Test user parameter adjustments at the HITL interruption points."""

    @pytest.mark.asyncio
    async def test_adjust_params(self, test_graph, graph_config, sample_state):
        """Test that user parameter adjustments are correctly merged."""
        # Run to first interrupt
        try:
            await test_graph.ainvoke(sample_state, graph_config)
        except Exception:
            pass

        # Resume with adjusted parameters
        adjustments = {
            "action": "adjust",
            "adjustments": {
                "skin_smoothing": 1.0,  # User wants lighter smoothing
                "whitening": 3.0,        # User wants more whitening
            },
        }

        try:
            await test_graph.ainvoke(Command(resume=adjustments), graph_config)
        except Exception:
            pass

        state = await test_graph.aget_state(graph_config)
        final_params = state.values.get("final_params")

        assert final_params is not None
        assert final_params["skin_smoothing"] == 1.0  # User override
        assert final_params["whitening"] == 3.0         # User override


class TestMemory:
    """Test checkpointer persistence across multiple invocations."""

    @pytest.mark.asyncio
    async def test_session_persistence(self, test_graph, graph_config, sample_state):
        """Test that state is persisted in the checkpointer across interactions."""
        # Run to first interrupt
        try:
            await test_graph.ainvoke(sample_state, graph_config)
        except Exception:
            pass

        # Get state — should be persisted
        state1 = await test_graph.aget_state(graph_config)
        assert state1 is not None
        assert state1.values.get("session_id") == "test-session-001"

        # Resume and confirm
        try:
            await test_graph.ainvoke(Command(resume={"action": "confirm"}), graph_config)
        except Exception:
            pass

        state2 = await test_graph.aget_state(graph_config)
        assert state2.values.get("workflow_stage") == "beautified"

        # Complete with feedback
        final = await test_graph.ainvoke(
            Command(resume={"satisfaction_score": 5}),
            graph_config,
        )
        assert final["workflow_stage"] == "completed"


class TestErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_invalid_base64(self, test_graph, graph_config):
        """Test that invalid base64 input is caught."""
        bad_state = {
            "session_id": "test-bad-001",
            "user_id": "test-user",
            "input_image_b64": "!!!not-valid-base64!!!",
            "workflow_stage": "start",
            "retry_count": 0,
        }

        result = await test_graph.ainvoke(bad_state, graph_config)
        assert result["workflow_stage"] == "input_failed"
        assert "Invalid base64" in result.get("error_message", "")


class TestStoreOperations:
    """Test long-term memory Store operations."""

    @pytest.mark.asyncio
    async def test_store_access(self, test_graph):
        """Test that the store is accessible and writable."""
        store = test_graph.store

        # Write a test preference
        test_prefs = {"skin_smoothing": 2.5, "whitening": 1.5}
        await store.aput(("users", "test-user", "preferences"), "current", test_prefs)

        # Read it back
        item = await store.aget(("users", "test-user", "preferences"), "current")
        assert item is not None
        assert item.value["skin_smoothing"] == 2.5


class TestBridgeIntegration:
    """Test that LangGraph nodes correctly call the bridge."""

    @pytest.mark.asyncio
    async def test_bridge_called_for_analysis(self, test_graph, graph_config, sample_state):
        """Test that the analyze_image node calls the bridge."""
        # The mock orchestrator tracks calls
        bridge = graph_config["configurable"]["bridge"]
        mock_orch = bridge._orchestrator

        # Run to first interrupt
        try:
            await test_graph.ainvoke(sample_state, graph_config)
        except Exception:
            pass

        # Verify analysis was delegated
        assert len(mock_orch.analysis_calls) >= 1, \
            "Expected at least one analysis delegation call"
        call = mock_orch.analysis_calls[0]
        assert call["session_id"] == "test-session-001"
        assert call["user_id"] == "test-user-001"

    @pytest.mark.asyncio
    async def test_bridge_called_for_beautification(self, test_graph, graph_config, sample_state):
        """Test that the apply_beautification node calls the bridge."""
        bridge = graph_config["configurable"]["bridge"]
        mock_orch = bridge._orchestrator

        # Run to first interrupt
        try:
            await test_graph.ainvoke(sample_state, graph_config)
        except Exception:
            pass

        # Resume with confirm
        try:
            await test_graph.ainvoke(Command(resume={"action": "confirm"}), graph_config)
        except Exception:
            pass

        # Verify beautification was delegated
        assert len(mock_orch.beautification_calls) >= 1, \
            "Expected at least one beautification delegation call"
