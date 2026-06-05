"""
MCP tool provider for the beautification model.

Wraps the beautification model HTTP client as an MCP-style tool.
Converts structured parameters to natural language descriptions
for the beautification model.
"""

from agent_face.langgraph_brain.state import BeautifyParams
from agent_face.models.beautify_client import BeautifyModelClient


class BeautifyMCPTool:
    """
    MCP-compatible tool wrapping the beautification model.

    Tool name: "beautify_face_image"
    Description: "Apply beautification enhancements to a facial image"

    This tool is registered with MAF's tool registry so that
    the BeautificationAgent can discover and invoke it via MCP.
    """

    name = "beautify_face_image"
    description = (
        "Apply beautification enhancements to a facial image. "
        "Takes an image and beautification parameters (0.0-5.0 scale) "
        "and returns the enhanced image. Parameters include skin smoothing, "
        "whitening, eye enlargement, face slimming, blush, lip color, "
        "blemish removal, nose reshaping, and eyebrow adjustment."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "image_b64": {
                "type": "string",
                "description": "Base64-encoded facial image to beautify",
            },
            "params": {
                "type": "object",
                "description": "Beautification parameters (0.0-5.0 per dimension)",
                "properties": {
                    "skin_smoothing": {"type": "number"},
                    "whitening": {"type": "number"},
                    "eye_enlargement": {"type": "number"},
                    "face_slimming": {"type": "number"},
                    "blush": {"type": "number"},
                    "lip_color_adjustment": {"type": "number"},
                    "blemish_removal": {"type": "number"},
                    "nose_reshaping": {"type": "number"},
                    "eyebrow_adjustment": {"type": "number"},
                },
            },
        },
        "required": ["image_b64", "params"],
    }

    def __init__(self):
        self._client = BeautifyModelClient()

    async def execute(self, image_b64: str, params: BeautifyParams) -> str:
        """
        Execute the tool — calls the beautification model.

        Args:
            image_b64: Base64-encoded facial image.
            params: Beautification parameters to apply.

        Returns:
            Base64-encoded beautified image.
        """
        return await self._client.beautify(
            image_b64=image_b64,
            params=params,
        )

    async def health_check(self) -> dict:
        """Check if the underlying model service is reachable."""
        return await self._client.health_check()
