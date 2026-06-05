"""
MultimodalAgent — MAF agent that analyzes facial images.

This agent wraps the multimodal MCP tool and is responsible for:
1. Receiving analysis tasks from the Orchestrator via A2A
2. Calling the multimodal model through the MCP tool
3. Returning structured AnalysisResult
"""

from typing import Optional
from agent_face.langgraph_brain.state import AnalysisResult, BeautifyParams
from agent_face.maf_body.tools.multimodal_tool import MultimodalMCPTool


class MultimodalAgent:
    """
    MAF agent for image analysis via the multimodal model.

    Agent type: specialist
    A2A agent card: multimodal.yaml
    MCP tools: [analyze_face_image]
    """

    def __init__(self, tool: Optional[MultimodalMCPTool] = None):
        self._tool = tool or MultimodalMCPTool()

    async def analyze(
        self,
        image_b64: str,
        prompt: Optional[str] = None,
        preferences: Optional[BeautifyParams] = None,
        session_count: int = 0,
        avg_satisfaction: float = 0.0,
    ) -> AnalysisResult:
        """
        Analyze a facial image and produce beautification suggestions.

        Called by the Orchestrator after input safety checks pass.

        Args:
            image_b64: Base64-encoded facial image.
            prompt: Optional natural language instruction.
            preferences: Optional user aesthetic preferences as hints.

        Returns:
            Structured AnalysisResult with suggested parameters.

        Raises:
            Exception: If the multimodal model call fails.
        """
        return await self._tool.execute(
            image_b64=image_b64,
            prompt=prompt,
            preferences=preferences,
            session_count=session_count,
            avg_satisfaction=avg_satisfaction,
        )

    async def health_check(self) -> dict:
        """Check agent and underlying tool health."""
        return await self._tool.health_check()
