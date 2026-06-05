"""
MCP tool provider for the multimodal model.

Wraps the multimodal model HTTP client as an MCP-style tool
that MAF agents can invoke. Uses MCP's StreamableHTTP transport
pattern for tool discovery and invocation.
"""

from typing import Optional
from agent_face.langgraph_brain.state import AnalysisResult, BeautifyParams
from agent_face.models.multimodal_client import MultimodalModelClient


class MultimodalMCPTool:
    """
    MCP-compatible tool wrapping the multimodal model.

    Tool name: "analyze_face_image"
    Description: "Analyze a facial image and suggest beautification parameters"

    This tool is registered with MAF's tool registry so that
    the MultimodalAgent can discover and invoke it via MCP.
    """

    name = "analyze_face_image"
    description = (
        "Analyze a facial image using the multimodal vision model. "
        "Returns skin tone, skin condition, detected facial features, "
        "detected skin issues, and suggested beautification parameters "
        "on a 0.0-5.0 scale for each dimension."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "image_b64": {
                "type": "string",
                "description": "Base64-encoded image of a human face",
            },
            "prompt": {
                "type": "string",
                "description": "Optional natural language instruction for the analysis",
            },
            "preferences": {
                "type": "object",
                "description": "Optional user preference parameters to use as hints",
            },
        },
        "required": ["image_b64"],
    }

    def __init__(self):
        self._client = MultimodalModelClient()

    async def execute(
        self,
        image_b64: str,
        prompt: Optional[str] = None,
        preferences: Optional[BeautifyParams] = None,
        session_count: int = 0,
        avg_satisfaction: float = 0.0,
    ) -> AnalysisResult:
        return await self._client.analyze(
            image_b64=image_b64,
            prompt=prompt,
            preferences=preferences,
            session_count=session_count,
            avg_satisfaction=avg_satisfaction,
        )

    async def health_check(self) -> dict:
        """Check if the underlying model service is reachable."""
        return await self._client.health_check()
