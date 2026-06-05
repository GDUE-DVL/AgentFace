"""
Multimodal Model Service — XiaoMi MIMO API integration.

Acts as an adapter between AgentFace's internal /analyze protocol
and MIMO's OpenAI-compatible Chat Completions API.

This service is deployed on port 8001 and called by
multimodal_client.py via the MAF MCP tool.

MIMO model: mimo-v2-omni (vision-capable multimodal model)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(_env_path)

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal Model Service", version="2.0.0")

# ── Model API Config ──────────────────────────────────────────
# Supports any OpenAI-compatible API (MIMO, vLLM, local server, etc.)

LLM_API_KEY = os.getenv("LLM_API_KEY", "no")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://210.38.72.13:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-31b-it-ms")

# ── Request/Response Models ────────────────────────────────────


class AnalyzeRequest(BaseModel):
    image_b64: str
    prompt: Optional[str] = None
    preferences: Optional[dict] = None


class SuggestedParams(BaseModel):
    skin_smoothing: float = 2.0
    whitening: float = 1.0
    eye_enlargement: float = 1.5
    face_slimming: float = 1.0
    blush: float = 1.0
    lip_color_adjustment: float = 1.0
    blemish_removal: float = 3.0
    nose_reshaping: float = 0.5
    eyebrow_adjustment: float = 1.0


class AnalyzeResponse(BaseModel):
    skin_tone: str
    skin_condition: str
    detected_features: list[str]
    detected_issues: list[str]
    suggested_params: SuggestedParams
    reasoning: str
    confidence: float


# ── Prompt Template ────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """你是一个专业的人像分析和美颜顾问。

## 第一步：判断是否为人脸照片
首先判断图片中是否包含清晰的人脸。
- 如果图片中没有人脸（如：风景、动物、物体、纯色图、文字截图等），is_face 设为 false，confidence 设为 0.0，reasoning 说明"未检测到人脸"，其他字段随意填写但不会被使用。
- 如果图片包含人脸，is_face 设为 true，进行第二步分析。

## 第二步：分析人像并给出美颜建议
请严格按照以下 JSON 格式回复（不要包含其他文字，只输出 JSON）：

{
  "is_face": true或false,
  "skin_tone": "肤色（白皙/自然/小麦/偏黄/偏暗）",
  "skin_condition": "皮肤状况（良好/有痘痘/有斑点/毛孔粗大/有黑眼圈等）",
  "detected_features": ["面部特征（双眼皮/高鼻梁/瓜子脸/圆脸等）"],
  "detected_issues": ["皮肤问题（黑眼圈/痘痘/色斑/肤色不均等）"],
  "suggested_params": {
    "skin_smoothing": 0.0-5.0,
    "whitening": 0.0-5.0,
    "eye_enlargement": 0.0-5.0,
    "face_slimming": 0.0-5.0,
    "blush": 0.0-5.0,
    "lip_color_adjustment": 0.0-5.0,
    "blemish_removal": 0.0-5.0,
    "nose_reshaping": 0.0-5.0,
    "eyebrow_adjustment": 0.0-5.0
  },
  "reasoning": "美颜建议理由（若非人脸，写'未检测到清晰人脸，请上传人像照片'）",
  "confidence": 0.0-1.0（非人脸为0.0）
}

## 美颜参数说明
- 0.0 = 不做处理
- 1.0-2.0 = 轻度处理（自然效果）
- 2.0-3.0 = 中度处理（可见效果）
- 3.0-4.0 = 较明显处理
- 4.0-5.0 = 重度处理

## 重要原则
1. 保持自然真实感，不过度美化
2. 根据实际皮肤状况给出合理建议
3. 肤色保持自然，不要盲目追求美白
4. 如果是男性，减少腮红和唇色调整
5. 优先处理明显的皮肤问题（痘痘、斑点等）"""


def build_user_prompt(prompt: Optional[str], preferences: Optional[dict]) -> str:
    """Build the user prompt for MIMO."""
    parts = ["请分析这张人像照片，给出美颜建议。"]
    if prompt:
        parts.append(f"用户要求：{prompt}")
    if preferences:
        prefs_str = ", ".join(f"{k}={v}" for k, v in preferences.items())
        parts.append(f"用户历史偏好参数（仅供参考）：{prefs_str}")
    return "\n".join(parts)


# ── Routes ─────────────────────────────────────────────────────


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a facial image using MIMO multimodal model.

    Sends the image + analysis prompt to MIMO API and parses
    the structured beautification recommendations.
    """
    logger.info("Analyzing image with MIMO API...")

    user_text = build_user_prompt(request.prompt, request.preferences)

    # Build OpenAI-compatible vision message
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{request.image_b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
            },
        )

        if response.status_code != 200:
            logger.error(f"MIMO API error: {response.status_code} {response.text}")
            raise ValueError(f"LLM API returned {response.status_code}: {response.text}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]

    logger.info(f"MIMO raw response: {content[:200]}...")

    # Parse JSON from response (handle markdown code blocks)
    parsed = _parse_analysis_json(content)

    is_face = parsed.get("is_face", False)
    sp = parsed.get("suggested_params") or {}  # handle null/None

    if not is_face:
        return AnalyzeResponse(
            skin_tone="—",
            skin_condition="—",
            detected_features=[],
            detected_issues=["未检测到人脸"],
            suggested_params=SuggestedParams(),  # all zeros
            reasoning=parsed.get("reasoning", "未检测到清晰人脸，请上传人像照片"),
            confidence=0.0,
        )

    return AnalyzeResponse(
        skin_tone=parsed.get("skin_tone", "—"),
        skin_condition=parsed.get("skin_condition", "—"),
        detected_features=parsed.get("detected_features", []),
        detected_issues=parsed.get("detected_issues", []),
        suggested_params=SuggestedParams(
            skin_smoothing=sp.get("skin_smoothing", 2.0),
            whitening=sp.get("whitening", 1.0),
            eye_enlargement=sp.get("eye_enlargement", 1.5),
            face_slimming=sp.get("face_slimming", 1.0),
            blush=sp.get("blush", 1.0),
            lip_color_adjustment=sp.get("lip_color_adjustment", 1.0),
            blemish_removal=sp.get("blemish_removal", 3.0),
            nose_reshaping=sp.get("nose_reshaping", 0.5),
            eyebrow_adjustment=sp.get("eyebrow_adjustment", 1.0),
        ),
        reasoning=parsed.get("reasoning", ""),
        confidence=parsed.get("confidence", 0.5),
    )


@app.get("/health")
async def health():
    """Health check."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{LLM_BASE_URL}/models",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
            return {"status": "ok", "mimo_api": resp.status_code == 200}
    except Exception as e:
        return {"status": "degraded", "mimo_api": str(e)}


def _parse_analysis_json(content: str) -> dict:
    """Parse JSON from MIMO response, handling markdown code blocks."""
    # Strip markdown code fences if present
    text = content.strip()
    if text.startswith("```"):
        # Find first newline after opening ```
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        # Remove closing ```
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        import re
        json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from: {content}")
        return {}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
