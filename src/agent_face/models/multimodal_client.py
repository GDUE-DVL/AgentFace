"""
MIMO multimodal client — direct API integration.

Calls XiaoMi MIMO API (OpenAI-compatible) directly, no proxy.
"""

import json
import logging
import os
import re
from typing import Optional

import httpx

from agent_face.config import settings
from agent_face.langgraph_brain.state import AnalysisResult, BeautifyParams, BEAUTIFY_PARAM_LABELS, DEFAULT_BEAUTIFY_PARAMS

logger = logging.getLogger(__name__)

# The primary Qwen analysis already returns accessory inventory, defect checks,
# normalized boxes, and English P2P prompts.  Extra serial audits are kept
# behind this switch for debugging but disabled for the normal fast path.
SINGLE_ANALYSIS = os.environ.get("AGENT_SINGLE_ANALYSIS", "1").lower() in {"1", "true", "yes", "on"}

class MultimodalModelClient:
    """Direct MIMO API client — preference-driven analysis."""

    def __init__(self):
        self._base_url = settings.mimo_base_url.rstrip("/")
        self._api_key = settings.mimo_api_key
        self._model = settings.mimo_model
        self._timeout = settings.model_request_timeout

    @staticmethod
    def _has_cjk(text: str) -> bool:
        return bool(text and re.search(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))

    @staticmethod
    def _looks_like_short_identity(text: str) -> bool:
        if not text or MultimodalModelClient._has_cjk(text):
            return False
        if len(re.findall(r"[A-Za-z]+", text)) > 24:
            return False
        return not bool(re.search(
            r'\b(the person|in the image|background|scene|lighting|photo|smiling|'
            r'shirt|dress|top|jacket|clothing|outfit)\b',
            text.lower(),
        ))

    @staticmethod
    def _clean_english(text: str) -> str:
        text = (text or "").strip().strip('`"\'')
        if not text or MultimodalModelClient._has_cjk(text):
            return ""
        text = text.splitlines()[0].strip()
        text = re.sub(r'^(?:identity\s*:\s*|description\s*:\s*)', '', text, flags=re.I)
        return text.strip(' .')

    @staticmethod
    def _remove_retouchable_marks(text: str) -> str:
        """Remove skin defects from the target identity phrase.

        A scar/acne/wrinkle belongs in the *source* description so P2P knows what
        to edit, but it must not remain in the target identity or the diffusion
        model will faithfully preserve it.
        """
        value = text or ""
        location = r"(?:\s+(?:on|near|under|beside)\s+(?:the\s+)?[a-z]+(?:\s+[a-z]+){0,3})?"
        patterns = [
            rf"(?:,?\s*(?:with|and)\s+)?(?:a\s+)?(?:visible\s+)?(?:facial\s+)?scar{location}",
            rf"(?:,?\s*(?:with|and)\s+)?(?:visible\s+)?(?:facial\s+)?(?:acne|blemishes|pimples|acne\s+marks){location}",
            rf"(?:,?\s*(?:with|and)\s+)?(?:visible\s+)?(?:facial\s+)?(?:wrinkles|fine\s+lines){location}",
        ]
        for pattern in patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s{2,}", " ", value).replace(" ,", ",")
        value = re.sub(r"\s+(?:and|with)\s*$", "", value, flags=re.IGNORECASE)
        return value.strip(" ,.")

    @staticmethod
    def _normalize_bbox(value: object) -> list[float]:
        """Validate a model-provided normalized [x1,y1,x2,y2] box."""
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return []
        try:
            box = [max(0.0, min(1.0, float(v))) for v in value]
        except (TypeError, ValueError):
            return []
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 0.00005:
            return []
        return [x1, y1, x2, y2]

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
        """Build system prompt: read image + user prompt → output P2P prompt pair.

        The model outputs a pair of English prompts (source/target) that drive
        the H-Edit beautification model directly — no numeric param tiers.
        """
        base = """You are a professional portrait beauty consultant. Your job is to look at a face photo,
read the user's beautification request, and produce a PROMPT-TO-PROMPT (P2P) pair:
a source_description (the face AS-IS in the photo) and a target_description (the SAME face
after beautification, following the user's request).

## Two separate kinds of visual information (do not mix them):
### A. Identity to preserve exactly
Describe visible gender, approximate age, hair style + color, face geometry, the
current expression/mouth state, and distinctive accessories or permanent non-skin items (glasses, earrings, hat,
headband, hair clip, jewelry, tattoo, or birthmark). Include an item only when
clearly visible. Before writing the prompts, make a complete inventory of every
clearly visible accessory (for example each earring/hoop/stud, necklace, glasses,
hat, headband, hair clip, or jewelry), including its type and color when visible.
Never omit an accessory because it is small or partly occluded. These features
must remain in BOTH prompts, word-for-word, and must not be recolored, removed,
replaced, or newly invented. Describe expression_state literally and briefly,
especially whether lips are closed/open/pursed and whether teeth are visible.
This is preservation metadata, never a request to beautify or reshape the mouth.

### B. Skin issues to inspect and RETOUCH
Before writing either prompt, perform a mandatory issue audit over the forehead,
eyebrows, eyelids, nose, cheeks, lips, mouth corners, chin, and jaw. Judge each
issue independently as present or absent:
- scar: a clear localized persistent line or texture/color change;
- acne_or_blemishes: visible pimples, acne marks, or distinct spots;
- wrinkles_or_fine_lines: visible age lines or creases.
For every present defect, localize it anatomically and spatially before writing
the prompts. Inspect the mouth corners, lower lip, nasolabial area, cheeks, chin,
and jaw separately. A mark beside the mouth must be called a mouth corner/lip
mark or cheek mark when appropriate; do not call it chin merely because it is
below the cheek. When an issue is present, its normalized bbox must tightly cover
the visible defect (not the whole face) and must never be left empty.
Do not call pores, normal texture, facial shadows, lighting, or compression artifacts
an issue. A present issue MUST be named in source_skin with its approximate location
when visible. An absent issue MUST NOT be mentioned anywhere.

Scars, acne/blemishes, and wrinkles are RETOUCHABLE skin defects, not immutable
identity. If present, keep them in source_description so the editor knows exactly
what to fix, but REMOVE or soften them in target_description. Never copy a scar,
acne, or wrinkle phrase into target identity. Preserve facial structure, hair,
accessories, expression, pose, and background while changing only the skin defects.

## How to write the prompt pair (MUST both be ENGLISH, and SHORT):
The beauty model inverts the image using source_description and edits using the token
difference between the pair. LONG descriptions (clothing, background, scenery) dilute the
edit tokens and weaken the effect — so keep prompts MINIMAL:
- identity_features: the complete visible identity phrase (gender/age/hair color,
  face geometry, accessories and permanent non-skin items), ending before skin,
  e.g. 'a young woman with long black hair and round glasses'
- source_skin: 'natural skin' plus ONLY confirmed visible issues and locations.
 - target_skin: short, positive, location-specific skin wording. Avoid the broad
   phrase 'smooth bright skin' when a local issue is being fixed. Use 'clear intact
   skin on the chin', 'clear skin at the right mouth corner', or 'softly reduced
   fine lines' instead of a vague global phrase. For a scar, keep the source wording 'visible scar at [exact location]' so the defect is identified, and make the target an explicit local action: 'smooth natural skin at [exact location]; remove the scar completely at [exact location]; preserve the exact surrounding structure'. H-Edit responds better to this direct action than to a generic 'smooth skin' phrase.
 - When a scar touches a lip or mouth corner, make local scar removal the ONLY edit goal in target_skin. Say 'remove the scar completely at [exact location]' and explicitly preserve the observed mouth opening, teeth visibility, lip contour, and expression word-for-word. Do not add lip reshaping, smile, or global face changes.
- source_description = identity_features + source_skin.
- target_description = the same PRESERVED identity_features + target_skin, but with
  retouchable issue words removed from identity and explicit removal/softening actions
  in target_skin. The target must never describe a scar as still present.
- NEVER mention clothing, background, scenery, or lighting. Put only the short,
  literal current expression/mouth state in expression_state.
- THE MAIN BEAUTIFICATION GOAL IS SKIN: smooth, even, flawless, radiant.
  target_skin MUST ALWAYS contain the word "smooth" when the user asks for
  whitening/smoothing/blemish removal (or nothing specific) — e.g. "smooth glowing skin",
  "smooth bright skin", "smooth clear skin". NEVER output a skin phrase without "smooth".
- ONLY the skin part may change between source and target. Do NOT change brightness,
  color tone, clothing, background, hair, or any identity feature.
- Adapt details to the user's words: whitening → brighter/whiter + smooth;
  smoothing → smooth even skin; blemish removal → smooth clear clean skin;
  slimming → slimmer face. If no explicit request, suggest a natural subtle retouch.
- Male subjects: keep adjustments minimal and natural.
- All descriptions must be ONE short sentence each, no Chinese.

Output ONLY valid JSON (no markdown, no extra text). reasoning in CHINESE, descriptions in ENGLISH:

{
  "is_face": true,
  "identity_features": "ENGLISH complete visible identity phrase only, no skin (gender/age/hair color + accessories/marks)",
  "accessory_items": ["short English noun phrase for each clearly visible accessory, or []"],
  "expression_state": "short ENGLISH literal state, e.g. neutral expression with pursed closed lips and no teeth visible",
  "source_skin": "ENGLISH short skin phrase for the current face",
  "target_skin": "ENGLISH short skin phrase for the beautified face",
   "source_description": "identity_features + source_skin, one sentence",
"target_description": "same preserved identity + target_skin with confirmed defects removed, one sentence",
"scar_check": "present" or "absent",
"scar_location": "short English location if present, otherwise empty string",
"scar_bbox": [x1, y1, x2, y2] normalized 0-1 if present, otherwise [],
"acne_check": "present" or "absent",
"acne_bbox": [x1, y1, x2, y2] normalized 0-1 if present, otherwise [],
"wrinkle_check": "present" or "absent",
"wrinkle_bbox": [x1, y1, x2, y2] normalized 0-1 if present, otherwise [],
   "reasoning": "用中文简述理由",
  "confidence": 0.0-1.0
}"""
        return base

    @staticmethod
    def _build_user_prompt(
        prompt: Optional[str],
        preferences: Optional[dict],
        strength: dict,
    ) -> str:
        """Build user message: image + user's beautification request."""
        parts = [
            "Look at this portrait photo. Analyze the face and output the P2P prompt pair "
            "(source_description + target_description) as instructed.",
        ]
        if prompt:
            parts.append(f"User's beautification request: {prompt}")
            parts.append("Adapt target_description to this request.")
        else:
            parts.append("No explicit request: suggest a natural subtle beauty retouch.")

        if preferences and strength["level"] != "none":
            lines = ["", "User's aesthetic taste from past sessions (reference, not rigid):"]
            for key, label in BEAUTIFY_PARAM_LABELS.items():
                v = preferences.get(key, 0)
                if v > 0:
                    lines.append(f"  Likes {label} around {v:.1f}/5.0")
            parts.extend(lines)
            parts.append("\nUse preferences as taste direction, but adjust for this photo's actual face and the user's current request.")

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
                    "max_completion_tokens": 2048,
                    "temperature": 0.0,
                    "thinking": {"type": "disabled"},
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
                suggested_params=dict(DEFAULT_BEAUTIFY_PARAMS),
                reasoning="MIMO 返回了非预期的响应格式",
                confidence=0.0,
                source_description="",
                target_description="",
            )
        logger.info(f"MIMO raw: {content[:200]}...")
        parsed = self._parse_json(content)

        # Run a separate, narrowly scoped accessory audit.  The main beauty
        # response can focus on skin and accidentally omit a small earring or
        # headband; this second pass supplies only clearly visible accessories
        # and is merged into both P2P prompts below.
        accessory_audit = []
        if not SINGLE_ANALYSIS:
            accessory_audit = await self._accessory_audit(image_b64)
            if accessory_audit:
                parsed["accessory_items"] = accessory_audit

        # Run a focused, low-token visual audit for scars.  General beauty
        # descriptions often overlook a small mark even when the model can see
        # it when asked directly.  The audit is advisory and falls back cleanly
        # if a backend does not support the extra request.
        primary_scar_check = str(parsed.get("scar_check", "")).strip().lower()
        primary_reasoning = str(parsed.get("reasoning", "") or "").lower()
        scar_audit = await self._scar_audit(image_b64) if not SINGLE_ANALYSIS else {}
        if scar_audit:
            audit_scar_check = str(scar_audit.get("scar_check", "")).strip().lower()
            # Resolve disagreement conservatively: a clear positive from either
            # pass wins, so an isolated audit miss cannot erase a visible scar.
            primary_positive = primary_scar_check in {"present", "yes", "true", "有"}
            reasoning_positive = ("scar" in primary_reasoning or "疤痕" in primary_reasoning) and not (
                "no scar" in primary_reasoning or "没有疤痕" in primary_reasoning
            )
            audit_positive = audit_scar_check in {"present", "yes", "true", "有"}
            parsed["scar_check"] = "present" if (primary_positive or reasoning_positive or audit_positive) else audit_scar_check
            parsed["scar_location"] = scar_audit.get("scar_location", "") or parsed.get("scar_location", "")
            parsed["scar_bbox"] = scar_audit.get("scar_bbox", []) or self._normalize_bbox(parsed.get("scar_bbox"))
            # Apply the same conservative union to acne and wrinkles.  The
            # primary analysis may see a blemish that the short audit misses.
            for field in ("acne_check", "wrinkle_check"):
                primary_flag = str(parsed.get(field, "")).strip().lower()
                audit_flag = str(scar_audit.get(field, "")).strip().lower()
                if primary_flag in {"present", "yes", "true", "有"} or audit_flag in {"present", "yes", "true", "有"}:
                    parsed[field] = "present"
                else:
                    parsed[field] = audit_flag or primary_flag
            parsed["acne_bbox"] = scar_audit.get("acne_bbox", []) or self._normalize_bbox(parsed.get("acne_bbox"))
            parsed["wrinkle_bbox"] = scar_audit.get("wrinkle_bbox", []) or self._normalize_bbox(parsed.get("wrinkle_bbox"))

        # Pass only confirmed defect boxes downstream.  An empty list is
        # intentional: it keeps the original full-face HEdit path unchanged.
        edit_regions = []
        for kind, check_field, bbox_field in (
            ("scar", "scar_check", "scar_bbox"),
            ("acne", "acne_check", "acne_bbox"),
            ("wrinkle", "wrinkle_check", "wrinkle_bbox"),
        ):
            flag = str(parsed.get(check_field, "")).strip().lower()
            if flag in {"present", "yes", "true", "有"}:
                bbox = self._normalize_bbox(parsed.get(bbox_field))
                if bbox:
                    edit_regions.append({"type": kind, "bbox": bbox})
        parsed["edit_regions"] = edit_regions

        is_face = parsed.get("is_face", False)

        if not is_face:
            return AnalysisResult(
                skin_tone="—",
                skin_condition="—",
                detected_features=[],
                detected_issues=["未检测到人脸"],
                lighting="—",
                suggested_params=dict(DEFAULT_BEAUTIFY_PARAMS),
                reasoning=parsed.get("reasoning", "未检测到清晰人脸"),
                confidence=0.0,
                source_description="",
                target_description="",
            )

        src = (parsed.get("source_description", "") or "").strip()
        tgt = (parsed.get("target_description", "") or "").strip()
        identity = (parsed.get("identity_features") or "").strip()
        src_skin = (parsed.get("source_skin") or "").strip()
        tgt_skin = (parsed.get("target_skin") or "").strip()
        # Keep the scar decision explicit and enforce the model's own judgment in
        # the identity phrase that is passed to H-Edit.  This prevents a visible
        # scar from being silently dropped, while also preventing the word "scar"
        # from leaking into prompts when the audit says it is absent.
        scar_check = str(parsed.get("scar_check", "")).strip().lower()
        scar_location = self._clean_english(str(parsed.get("scar_location", "") or "")).strip()

        if not self._looks_like_short_identity(identity):
            identity = ""
        if not identity and not SINGLE_ANALYSIS:
            identity = await self._english_identity(src, image_b64)
        if not identity:
            # Single-analysis mode must not issue a second model request.  The
            # main response is instructed to provide identity_features in
            # English; if it fails validation, use a deterministic safe phrase.
            identity = "a person with natural hair"

        identity = self._merge_accessories(identity, parsed.get("accessory_items", []))

        # Keep all retouchable defects out of the immutable identity.  Add a
        # confirmed scar back to the source side only, where it acts as a precise
        # edit anchor; the target side intentionally omits it.
        identity_base = self._remove_retouchable_marks(identity)
        expression_state = self._clean_english(
            str(parsed.get("expression_state", "") or "")
        ).strip(" ,.")
        # Realistic SDXL fine-tunes have a strong smiling-portrait prior.  Carry
        # the observed mouth/expression state verbatim on both branches so skin
        # retouching cannot silently turn closed or pursed lips into a toothy
        # smile.  This is prompt-level identity preservation, not a mouth mask.
        if expression_state and len(re.findall(r"[A-Za-z]+", expression_state)) <= 14:
            identity_base = f"{identity_base.rstrip(' .,')}, {expression_state}"
        source_identity = identity_base
        target_identity = identity_base
        if scar_check in {"present", "yes", "有", "true"}:
            location = f" on the {scar_location}" if scar_location else ""
            source_identity = f"{identity_base.rstrip(' .,')}, with a visible facial scar{location}"

        # Do not pass mixed-language text or long scene descriptions to H-Edit.
        # Rebuild both prompts from one identity and short English skin phrases.
        src_skin = self._clean_english(src_skin) or "natural skin"
        tgt_skin = self._clean_english(tgt_skin) or "subtle smooth even natural skin"
        # Preserve only issues explicitly confirmed by the visual audit in the
        # source prompt.  This gives H-Edit an accurate as-is description without
        # hallucinating acne or wrinkles on a clean face.
        scar_present = scar_check in {"present", "yes", "有", "true"}
        confirmed_issues = []
        # Do not stack several semantic edits when a scar is present. Realistic
        # portrait checkpoints may respond to a broad bundle of skin requests by
        # redrawing the entire mouth/expression. A single localized positive
        # transition gives P2P a much smaller and safer token difference.
        if not scar_present and str(parsed.get("acne_check", "")).lower() in {"present", "yes", "true", "有"}:
            confirmed_issues.append("visible blemishes")
        if not scar_present and str(parsed.get("wrinkle_check", "")).lower() in {"present", "yes", "true", "有"}:
            confirmed_issues.append("visible fine lines")
        if scar_present:
            # The scar is already expressed once in source_identity. Remove
            # incidental fine-line/blemish wording from source_skin so the
            # source/target pair represents one precise edit only.
            src_skin = "natural skin"
        for issue in confirmed_issues:
            if issue not in src_skin.lower():
                src_skin = f"{src_skin} with {issue}"
        if "smooth" not in tgt_skin.lower() and not confirmed_issues:
            tgt_skin = "smooth " + tgt_skin
        treatments = []
        if scar_present:
            scar_area = f"the {scar_location}" if scar_location else "the localized scar area"
            tgt_skin = (
                f"smooth natural skin at {scar_area}; remove the scar completely at {scar_area}; "
                "preserve the exact mouth shape, lip contour, expression, and all surrounding facial structure"
            )
        if not scar_present and str(parsed.get("acne_check", "")).lower() in {"present", "yes", "true", "有"}:
            treatments.append("remove acne and blemishes; preserve natural texture")
        if not scar_present and str(parsed.get("wrinkle_check", "")).lower() in {"present", "yes", "true", "有"}:
            treatments.append("reduce fine lines; preserve expression")
        if treatments:
            tgt_skin = f"{tgt_skin}, {', '.join(treatments)}"
        preserve_clause = (
            "same identity, unchanged facial geometry, lip contour, mouth opening, "
            "teeth visibility, expression, gaze, and pose"
        )
        src = f"{source_identity}, {preserve_clause}, with {src_skin}"
        tgt = f"{target_identity}, {preserve_clause}, with {tgt_skin}"

        if False:
            # 强制身份特征 verbatim 保留（眼镜/帽子/耳环/发型不能被模型漏掉）：
            # 用 identity_features 重组，source/target 只在皮肤短语上不同。
            if tgt_skin:
                # 皮肤方向硬约束：target 皮肤短语必须含 "smooth"（丝滑/平整是核心编辑目标）。
                # 不赌模型——缺了就强制补上，编辑焦点锁定在面部皮肤而非亮度/色调。
                if "smooth" not in tgt_skin.lower():
                    tgt_skin = "smooth " + tgt_skin
                tgt = f"{identity}, with {tgt_skin}"
            if src_skin:
                src = f"{identity}, with {src_skin}"

        if not tgt:
            logger.warning("target_description missing from model output — falling back to source")
            tgt = src
        return AnalysisResult(
            skin_tone="—",
            skin_condition="—",
            detected_features=[],
            detected_issues=[],
            lighting="—",
            suggested_params=dict(DEFAULT_BEAUTIFY_PARAMS),
            reasoning=parsed.get("reasoning", ""),
            confidence=float(parsed.get("confidence", 0.5)),
            # ``src`` was rebuilt from the validated English identity and the
            # audited skin phrase above.  Do not send it through another model
            # call here: that second paraphrase can silently drop a scar, mark,
            # or other explicitly confirmed feature.
            source_description=src,
            target_description=tgt,
            edit_regions=parsed.get("edit_regions", []),
        )

    @staticmethod
    def _merge_accessories(identity: str, items: object) -> str:
        """Append audited visible accessories to the immutable identity phrase."""
        if not isinstance(items, (list, tuple)):
            return identity
        keywords = re.compile(
            r"\b(earring|earrings|hoop|stud|glasses|spectacles|necklace|"
            r"jewelry|jewellery|hat|headband|hair clip|hairpin|barrette|"
            r"tiara|crown|tattoo|birthmark)\b",
            re.IGNORECASE,
        )
        result = identity.strip(" ,.")
        existing = result.lower()
        additions = []
        for raw in items:
            item = MultimodalModelClient._clean_english(str(raw))
            if not item or len(re.findall(r"[A-Za-z]+", item)) > 8 or not keywords.search(item):
                continue
            if item.lower() in existing:
                continue
            additions.append(item)
            existing += " " + item.lower()
        if additions:
            result = f"{result}, with visible {', '.join(additions)}"
        return result

    async def _accessory_audit(self, image_b64: str) -> list[str]:
        """Return only clearly visible non-skin accessory noun phrases."""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": (
                "Inspect this portrait only for clearly visible accessories or permanent non-skin items. "
                "Check both ears, neck, eyes, head, and hair: earrings/hoops/studs, glasses, necklace, "
                "hat, headband, hair clips, jewelry, tattoo, or birthmark. Include each item only if it is "
                "actually visible; do not infer hidden items. Return ONLY valid JSON in this exact form: "
                "{\"accessories\":[\"short English noun phrase\"]}. Use [] when none are clearly visible. "
                "Do not include hair, clothing, skin, background, or facial expression."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}]
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self._model,
                        "messages": messages,
                        "max_completion_tokens": 96,
                        "temperature": 0.0,
                        "thinking": {"type": "disabled"},
                    },
                )
                resp.raise_for_status()
                result = self._parse_json(resp.json()["choices"][0]["message"]["content"])
            items = result.get("accessories", [])
            return list(items) if isinstance(items, list) else []
        except Exception as e:
            logger.warning(f"Accessory audit failed; keeping primary identity: {e}")
            return []

    async def _english_identity(self, src: str, image_b64: str) -> str:
        """Ensure source_description is English (H-Edit needs English)."""
        if self._looks_like_short_identity(src):
            return self._clean_english(src)

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "Look at this image. Write ONE short English sentence describing only the person's "
                     "gender, approximate age, hair style and color, face geometry, and clearly visible "
                     "accessories or permanent non-skin items such as glasses, earrings, hats, "
                     "headbands, hair clips, tattoos, birthmarks, or freckles. Do not include scars, "
                     "acne, blemishes, wrinkles, or fine lines; those are retouchable skin issues. "
                    "Do not mention clothing, background, lighting, or expression. Output only that sentence."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                }},
            ]},
        ]
        try:
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
                        "max_completion_tokens": 64,
                        "temperature": 0.0,
                        "thinking": {"type": "disabled"},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            eng = self._clean_english(data["choices"][0]["message"]["content"])
            if eng:
                return eng
            logger.warning(f"English identity fallback returned non-English: {eng!r}")
            return ""
        except Exception as e:
            logger.warning(f"English identity fallback failed: {e}")
            return ""

    async def _scar_audit(self, image_b64: str) -> dict:
        """Ask the multimodal backend for explicit visible skin-mark judgments."""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": (
                "Inspect this face photo carefully for three visible skin issues: "
                "(1) scar, meaning a clear localized persistent-looking line or "
                "texture/color change; (2) acne or blemishes, meaning visible active "
                "pimples or distinct spots; (3) wrinkles or fine lines. Check the "
                "forehead, cheeks, nose, eyelids, lips, mouth corners, chin, and jaw. "
                "Do not classify pores, normal skin texture, shadows, lighting, or "
                "compression artifacts as any issue. Return ONLY valid JSON: "
                "{\"scar_check\": \"present\" or \"absent\", "
                "\"scar_location\": \"short English location, or empty string\", "
                "\"scar_bbox\": [x1,y1,x2,y2] normalized 0-1 or [], "
                "\"acne_check\": \"present\" or \"absent\", "
                "\"acne_bbox\": [x1,y1,x2,y2] normalized 0-1 or [], "
                "\"wrinkle_check\": \"present\" or \"absent\", "
                "\"wrinkle_bbox\": [x1,y1,x2,y2] normalized 0-1 or [], "
                "\"evidence\": \"short reason\"}."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}]
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self._model,
                        "messages": messages,
                        "max_completion_tokens": 128,
                        "temperature": 0.0,
                        "thinking": {"type": "disabled"},
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
            result = self._parse_json(content)
            check = str(result.get("scar_check", "")).strip().lower()
            if check not in {"present", "absent", "yes", "no", "true", "false", "有", "没有"}:
                return {}
            if check in {"yes", "true", "有"}:
                check = "present"
            elif check in {"no", "false", "没有"}:
                check = "absent"
            location = str(result.get("scar_location", "") or "").strip()
            # Keep the downstream prompt English even if a backend answers the
            # location in Chinese.
            location_map = {
                "下巴": "chin", "下颌": "jaw", "脸颊": "cheek", "面颊": "cheek",
                "额头": "forehead", "鼻子": "nose", "嘴角": "mouth corner",
                "嘴唇": "lip", "下唇": "lower lip", "上唇": "upper lip",
            }
            for key, value in location_map.items():
                if key in location:
                    location = value
                    break
            if re.search(r"[\u4e00-\u9fff]", location):
                location = ""
            def normalize_flag(value: object) -> str:
                flag = str(value or "").strip().lower()
                if flag in {"present", "yes", "true", "有"}:
                    return "present"
                if flag in {"absent", "no", "false", "没有"}:
                    return "absent"
                return ""
            return {
                "scar_check": check,
                "scar_location": location,
                "scar_bbox": self._normalize_bbox(result.get("scar_bbox")),
                "acne_check": normalize_flag(result.get("acne_check")),
                "acne_bbox": self._normalize_bbox(result.get("acne_bbox")),
                "wrinkle_check": normalize_flag(result.get("wrinkle_check")),
                "wrinkle_bbox": self._normalize_bbox(result.get("wrinkle_bbox")),
            }
        except Exception as e:
            logger.warning(f"Scar audit failed; keeping primary analysis: {e}")
            return {}

    async def health_check(self) -> dict:
        """Check MIMO API connectivity."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "messages": [{"role":"user","content":"hi"}], "max_completion_tokens":5},
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

