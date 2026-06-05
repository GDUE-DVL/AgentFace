"""
LangGraph StateGraph builder.

Compiles the beautification workflow state machine with:
- 8 workflow nodes + 1 error handler
- Conditional routing edges
- Checkpointer for short-term memory (session state snapshots)
- Store for long-term memory (user preferences, history, feedback)

This is the "Brain" of AgentFace.
"""

import logging
from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

from agent_face.langgraph_brain.state import BeautifyWorkflowState
from agent_face.langgraph_brain.nodes.receive_input import receive_input
from agent_face.langgraph_brain.nodes.retrieve_context import retrieve_context
from agent_face.langgraph_brain.nodes.analyze_image import analyze_image
from agent_face.langgraph_brain.nodes.present_plan import present_plan
from agent_face.langgraph_brain.nodes.apply_beautification import apply_beautification
from agent_face.langgraph_brain.nodes.collect_feedback import collect_feedback
from agent_face.langgraph_brain.nodes.update_memory import update_memory
from agent_face.langgraph_brain.nodes.finalize import finalize
from agent_face.langgraph_brain.nodes.handle_error import handle_error
from agent_face.langgraph_brain.routing import (
    route_after_input,
    route_after_analysis,
    route_after_beautification,
    route_after_error,
)
from agent_face.bridge.maf_client import MAFBridgeClient

logger = logging.getLogger(__name__)


def build_graph(
    bridge: Optional[MAFBridgeClient] = None,
    checkpointer = None,
    store = None,
):
    """
    Build and compile the beautification workflow StateGraph.

    Args:
        bridge: MAFBridgeClient for calling MAF agents.
        checkpointer: LangGraph checkpointer. Defaults to MemorySaver.
        store: LangGraph store. Defaults to InMemoryStore.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()
    if store is None:
        store = InMemoryStore()

    # Build the graph
    builder = StateGraph(BeautifyWorkflowState)

    # ── Add all nodes ───────────────────────────────────────────
    builder.add_node("receive_input", receive_input)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("analyze_image", analyze_image)
    builder.add_node("present_plan", present_plan)
    builder.add_node("apply_beautification", apply_beautification)
    builder.add_node("collect_feedback", collect_feedback)
    builder.add_node("update_memory", update_memory)
    builder.add_node("finalize", finalize)
    builder.add_node("handle_error", handle_error)

    # ── Add edges (happy path) ──────────────────────────────────
    builder.add_edge(START, "receive_input")

    # Conditional: go to retrieve_context or handle_error
    builder.add_conditional_edges(
        "receive_input",
        route_after_input,
        {
            "retrieve_context": "retrieve_context",
            "handle_error": "handle_error",
        },
    )

    builder.add_edge("retrieve_context", "analyze_image")

    # Conditional: go to present_plan or handle_error
    builder.add_conditional_edges(
        "analyze_image",
        route_after_analysis,
        {
            "present_plan": "present_plan",
            "handle_error": "handle_error",
        },
    )

    # present_plan is a HITL interrupt — after resume, go to apply_beautification
    builder.add_edge("present_plan", "apply_beautification")

    # Conditional: go to collect_feedback or handle_error
    builder.add_conditional_edges(
        "apply_beautification",
        route_after_beautification,
        {
            "collect_feedback": "collect_feedback",
            "handle_error": "handle_error",
        },
    )

    # collect_feedback is a HITL interrupt — after resume, go to update_memory
    builder.add_edge("collect_feedback", "update_memory")
    builder.add_edge("update_memory", "finalize")
    builder.add_edge("finalize", END)

    # ── Error recovery path ─────────────────────────────────────
    builder.add_conditional_edges(
        "handle_error",
        route_after_error,
        {
            "analyze_image": "analyze_image",
            "apply_beautification": "apply_beautification",
            "finalize": "finalize",
        },
    )

    # ── Compile with checkpointer and store ─────────────────────
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    logger.info(
        "LangGraph compiled successfully: "
        f"9 nodes, checkpointer={type(checkpointer).__name__}, "
        f"store={type(store).__name__}"
    )

    return graph
