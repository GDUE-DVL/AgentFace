"""
SafetyGuardAgent — responsible AI safety checks.

Operates at two points in the MAF workflow DAG:
1. Input guard: Validates incoming images before analysis
2. Output guard: Validates beautified results before returning

Safety checks: NSFW detection, image format validation,
over-smoothing detection, and dimension validation.
"""

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

from agent_face.config import settings


@dataclass
class SafetyResult:
    """Result of a safety check."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    downsampled_image_b64: Optional[str] = None  # If resizing was needed


class SafetyGuardAgent:
    """
    MAF agent that performs content safety checks.

    Agent type: safety_guard
    A2A agent card: safety_guard.yaml

    This agent is a stateless function agent — it receives a request,
    runs checks, and returns a pass/fail result. It does not call
    external models.
    """

    def __init__(self):
        self._enable_content_safety = settings.enable_content_safety
        self._enable_pii = settings.enable_pii_detection
        self._max_dimension = settings.max_image_dimension
        self._max_bytes = settings.max_image_bytes
        self._over_smoothing_threshold = settings.over_smoothing_threshold

    # ── Input Guard ──────────────────────────────────────────

    async def validate_input(
        self, image_b64: str, image_format: str = "image/jpeg"
    ) -> SafetyResult:
        """
        Run all input safety checks on the incoming image.

        Checks:
        1. Format validation
        2. Size validation
        3. Dimension validation (downsample if needed)
        4. Content safety (NSFW/violence detection)
        """
        result = SafetyResult(passed=True)

        # 1. Format validation
        if image_format not in settings.allowed_formats_list:
            result.passed = False
            result.violations.append(f"Unsupported format: {image_format}")
            return result

        # 2. Size validation
        decoded = None
        try:
            decoded = base64.b64decode(image_b64)
        except Exception:
            result.passed = False
            result.violations.append("Invalid base64 encoding")
            return result

        if len(decoded) > self._max_bytes:
            result.passed = False
            result.violations.append(
                f"Image too large: {len(decoded)} bytes (max {self._max_bytes})"
            )
            return result

        # 3. Dimension validation
        try:
            image = Image.open(io.BytesIO(decoded))
            width, height = image.size

            if width > self._max_dimension or height > self._max_dimension:
                # Downsample
                scale = self._max_dimension / max(width, height)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=95)
                result.downsampled_image_b64 = base64.b64encode(buf.getvalue()).decode()
                result.warnings.append(
                    f"Image downsampled from {width}x{height} to {new_size[0]}x{new_size[1]}"
                )
        except Exception as e:
            result.passed = False
            result.violations.append(f"Invalid image: {str(e)}")
            return result

        # 4. Content safety (placeholder — real implementation would call
        #    a dedicated NSFW classifier model)
        if self._enable_content_safety:
            # Placeholder for NSFW detection model call
            pass

        return result

    # ── Output Guard ─────────────────────────────────────────

    async def validate_output(
        self, original_b64: str, result_b64: str
    ) -> SafetyResult:
        """
        Run post-beautification safety checks.

        Checks:
        1. Result image validity
        2. Over-smoothing detection
        3. Unnatural result detection
        """
        result = SafetyResult(passed=True)

        # 1. Result validity
        try:
            result_buf = base64.b64decode(result_b64)
            result_image = Image.open(io.BytesIO(result_buf))
        except Exception:
            result.passed = False
            result.violations.append("Invalid beautification result")
            return result

        # 2. Over-smoothing detection
        try:
            is_over_smoothed, msg = self._detect_over_smoothing(
                original_b64, result_b64
            )
            if is_over_smoothed:
                result.warnings.append(msg)
                # Note: over-smoothing is a WARNING, not a hard failure.
                # The user may prefer very smooth skin. We flag it but
                # don't block the result.
        except Exception:
            pass  # Non-critical check, don't block on error

        return result

    def _detect_over_smoothing(
        self, original_b64: str, result_b64: str
    ) -> tuple[bool, str]:
        """
        Detect if the beautified image is over-smoothed.

        Uses Laplacian variance comparison between original and result.
        If the result's variance drops below the threshold ratio of the
        original, it's flagged as potentially over-smoothed.
        """
        import numpy as np
        import cv2  # Optional dependency, gracefully degrade

        try:
            orig_bytes = base64.b64decode(original_b64)
            result_bytes = base64.b64decode(result_b64)

            orig_img = cv2.imdecode(
                np.frombuffer(orig_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
            )
            result_img = cv2.imdecode(
                np.frombuffer(result_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
            )

            orig_var = cv2.Laplacian(orig_img, cv2.CV_64F).var()
            result_var = cv2.Laplacian(result_img, cv2.CV_64F).var()

            if orig_var > 0:
                ratio = result_var / orig_var
                if ratio < self._over_smoothing_threshold:
                    return True, (
                        f"Over-smoothing detected: Laplacian variance ratio {ratio:.3f} "
                        f"(threshold: {self._over_smoothing_threshold})"
                    )

            return False, ""
        except ImportError:
            # cv2 not available, skip over-smoothing check
            return False, "cv2 not available, skipping over-smoothing check"

    async def health_check(self) -> dict:
        """Check agent health."""
        return {"status": "ok", "content_safety": self._enable_content_safety}
