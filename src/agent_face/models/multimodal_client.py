"""
MIMO multimodal client — direct API integration.

Calls XiaoMi MIMO API (OpenAI-compatible) directly, no proxy.
"""

import json
import logging
import re
from typing import Optional

import httpx

from agent_face.config import settings
from agent_face.langgraph_brain.state import AnalysisResult, BeautifyParams, BEAUTIFY_PARAM_LABELS

logger = logging.getLogger(__name__)

class MultimodalModelClient:
    """Direct MIMO API client — preference-driven analysis."""

    def __init__(self):
        self._base_url = settings.mimo_base_url.rstrip("/")
        self._api_key = settings.mimo_api_key
        self._model = settings.mimo_model
        self._timeout = settings.model_request_timeout

    # ── Preference confidence ──────────────────────────────────

    @staticmethod
    def _pref_strength(session_count: int, avg_satisfaction: float) -> dict:
        """Calculate how strongly to enforce user preferences.

        Returns {"level": "strong"|"moderate"|"weak", "range": ±tolerance}
        """
        if session_count >= 5 and avg_satisfaction >= 4.0:
            return {"level": "strong", "range": 0.5}    # ±0.5 of preference
        elif session_count >= 3 and avg_satisfaction >= 3.5:
            return {"level": "moderate", "range": 1.0}   # ±1.0
        elif session_count >= 1:
            return {"level": "weak", "range": 2.0}       # ±2.0
        else:
            return {"level": "none", "range": 5.0}       # free range

    @staticmethod
    def _build_system_prompt(strength: dict) -> str:
        """Build MIMO system prompt with holistic analysis."""
        base = """You are a professional portrait beauty consultant. Your job is to holistically analyze a face photo
and suggest optimal beautification parameters by considering ALL of the following together:

1. FACE ANALYSIS: face shape (oval/round/square/heart/long), facial features (eyes, nose, eyebrows, lips),
   facial proportions, gender, approximate age, expression
2. SKIN ANALYSIS: skin tone, texture, pores, wrinkles, blemishes, acne, dark circles, unevenness
3. LIGHTING ANALYSIS: is the photo warm-lit, cold-lit, dim, bright, studio, outdoor, mixed?
4. USER HISTORY: accumulated aesthetic preferences from past sessions (if available below)

## How to synthesize:

You MUST weigh all four factors together to produce a single set of parameters.
Do NOT treat user preferences as rigid ranges. Instead, use them as one input among others:

- Oval face + male → minimal face slimming, skip blush
- Round face + female → moderate face slimming may be appropriate
- Dim/warm light → reduce whitening (photo already has warm glow), increase smoothing (noise in shadows)
- Bright/cold light → moderate whitening OK, be careful not to over-smooth visible details
- User prefers high whitening BUT photo is warm-lit → compromise: whitening lower than preference
- User prefers low smoothing BUT skin has visible acne → increase smoothing to address actual issues
- Strong jaw + male → skip nose/eyebrow adjustments, preserve masculine features
- Deep-set eyes + dark circles → target blemish removal at eye area

## Key principles:
- Preserve natural facial identity and unique features
- Parameters should make sense for THIS specific photo, not blindly follow history
- Male subjects: minimal blush, lip color, eye enlargement
- Female subjects: all parameters available, keep natural look
- Explain your reasoning: mention which factors influenced each parameter decision

"""

        if strength["level"] in ("strong", "moderate"):
            base += f"""
This user has {strength['level']} preference history ({strength.get('sessions',0)} sessions).
Consider their preferences seriously, but still adjust based on this specific photo's conditions.
"""
        elif strength["level"] == "weak":
            base += "This user has limited history. Prioritize photo analysis over preferences.\n"

        base += """
Output ONLY valid JSON (no markdown, no extra text). Reply in CHINESE for all text fields:

{
  "is_face": true,
  "skin_tone": "肤色（如：白皙/自然/小麦/偏暖/偏黄）",
  "skin_condition": "用中文描述皮肤状况",
  "detected_features": ["用中文列出面部特征，包括脸型、五官特点、性别年龄估计"],
  "detected_issues": ["用中文列出皮肤问题"],
  "lighting": "光线状况（如：自然光/暖光/冷光/暗光/强光/混合光）",
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
  "reasoning": "用中文简述建议理由",
  "confidence": 0.0-1.0
}

Scale: 0=none 1-2=light 2-3=moderate 3-4=noticeable 4-5=heavy.
Keep natural look. Reduce blush/lip for males."""
        return base

    @staticmethod
    def _build_user_prompt(
        prompt: Optional[str],
        preferences: Optional[dict],
        strength: dict,
    ) -> str:
        """Build user message with preference directives."""
        parts = ["Analyze this portrait photo and suggest beautification parameters. Reply in Chinese."]
        if prompt:
            parts.append(f"User's current request: {prompt}")

        if preferences and strength["level"] != "none":
            lines = ["", "User's aesthetic taste from past sessions (reference, not rigid):"]
            for key, label in BEAUTIFY_PARAM_LABELS.items():
                v = preferences.get(key, 0)
                if v > 0:
                    lines.append(f"  Likes {label} around {v:.1f}/5.0")
            parts.extend(lines)
            parts.append("\nUse preferences as taste direction, but adjust for this photo's actual face shape, skin condition, and lighting.")

        return "\n".join(parts)

    async def analyze(
        self,
        image_b64: str,
        prompt: Optional[str] = None,
        preferences: Optional[BeautifyParams] = None,
        session_count: int = 0,
        avg_satisfaction: float = 0.0,
    ) -> AnalysisResult:
        """Analyze image via MIMO — preference-driven analysis.

        Preferences now directly constrain MIMO's output range.
        The stronger the user's preference history, the tighter the constraint.
        """
        strength = self._pref_strength(session_count, avg_satisfaction)
        system = self._build_system_prompt(strength)
        user_text = self._build_user_prompt(prompt, preferences, strength)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high",
                }},
            ]},
        ]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected MIMO response structure: {e}")
            return AnalysisResult(
                skin_tone="—",
                skin_condition="—",
                detected_features=[],
                detected_issues=["模型响应异常"],
                lighting="—",
                suggested_params={k: 0.0 for k in BeautifyParams.__optional_keys__()},
                reasoning="MIMO 返回了非预期的响应格式",
                confidence=0.0,
            )
        logger.info(f"MIMO raw: {content[:200]}...")
        parsed = self._parse_json(content)

        is_face = parsed.get("is_face", False)
        sp = parsed.get("suggested_params") or {}

        if not is_face:
            return AnalysisResult(
                skin_tone="—",
                skin_condition="—",
                detected_features=[],
                detected_issues=["未检测到人脸"],
                lighting="—",
                suggested_params={k: 0.0 for k in BeautifyParams.__optional_keys__()},
                reasoning=parsed.get("reasoning", "未检测到清晰人脸"),
                confidence=0.0,
            )

        return AnalysisResult(
            skin_tone=parsed.get("skin_tone", "—"),
            skin_condition=parsed.get("skin_condition", "—"),
            detected_features=parsed.get("detected_features", []),
            detected_issues=parsed.get("detected_issues", []),
            lighting=parsed.get("lighting", "—"),
            suggested_params=BeautifyParams(
                skin_smoothing=float(sp.get("skin_smoothing", 2.0)),
                whitening=float(sp.get("whitening", 1.0)),
                eye_enlargement=float(sp.get("eye_enlargement", 1.5)),
                face_slimming=float(sp.get("face_slimming", 1.0)),
                blush=float(sp.get("blush", 1.0)),
                lip_color_adjustment=float(sp.get("lip_color_adjustment", 1.0)),
                blemish_removal=float(sp.get("blemish_removal", 3.0)),
                nose_reshaping=float(sp.get("nose_reshaping", 0.5)),
                eyebrow_adjustment=float(sp.get("eyebrow_adjustment", 1.0)),
            ),
            reasoning=parsed.get("reasoning", ""),
            confidence=float(parsed.get("confidence", 0.5)),
        )

    async def health_check(self) -> dict:
        """Check MIMO API connectivity."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "messages": [{"role":"user","content":"hi"}], "max_tokens":5},
                )
                return {"status": "ok", "latency_ms": resp.elapsed.total_seconds() * 1000}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Parse JSON from model response, handling markdown fences."""
        text = content.strip()
        if text.startswith("```"):
            nl = text.find("\n")
            if nl != -1:
                text = text[nl+1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse JSON: {content[:200]}")
            return {}
