"""
FastAPI dependency injection.

Provides typed dependencies for LangGraph graph, bridge client,
and MAF orchestrator access in route handlers.
"""

from fastapi import Request
from langgraph.graph import StateGraph

from agent_face.bridge.maf_client import MAFBridgeClient
from agent_face.maf_body.orchestrator import MAFOrchestrator


def get_graph(request: Request) -> StateGraph:
    """Get the compiled LangGraph StateGraph from app state."""
    return request.app.state.graph


def get_bridge(request: Request) -> MAFBridgeClient:
    """Get the MAF bridge client from app state."""
    return request.app.state.bridge


def get_orchestrator(request: Request) -> MAFOrchestrator:
    """Get the MAF orchestrator from app state."""
    return request.app.state.orchestrator
