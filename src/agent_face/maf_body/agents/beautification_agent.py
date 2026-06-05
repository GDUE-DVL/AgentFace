"""
BeautificationAgent — MAF agent that applies beautification.

This agent wraps the beautification MCP tool and is responsible for:
1. Receiving beautification tasks from the Orchestrator via A2A
2. Converting structured params to natural language descriptions
3. Calling the beautification model through the MCP tool
4. Returning the enhanced image
"""

from typing import Optional
from agent_face.langgraph_brain.state import BeautifyParams
from agent_face.maf_body.tools.beautify_tool import BeautifyMCPTool


class BeautificationAgent:
    """
    MAF agent for image beautification via the beautification model.

    Agent type: specialist
    A2A agent card: beautification.yaml
    MCP tools: [beautify_face_image]
    """

    def __init__(self, tool: Optional[BeautifyMCPTool] = None):
        self._tool = tool or BeautifyMCPTool()

    async def beautify(
        self, image_b64: str, params: BeautifyParams
    ) -> str:
        """
        Apply beautification to a facial image.

        Called by the Orchestrator after the user confirms the plan.

        Args:
            image_b64: Base64-encoded facial image.
            params: Beautification parameters to apply.

        Returns:
            Base64-encoded beautified image.

        Raises:
            Exception: If the beautification model call fails.
        """
        return await self._tool.execute(
            image_b64=image_b64,
            params=params,
        )

    async def health_check(self) -> dict:
        """Check agent and underlying tool health."""
        return await self._tool.health_check()
