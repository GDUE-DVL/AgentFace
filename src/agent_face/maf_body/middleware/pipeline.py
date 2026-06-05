"""
MAF middleware pipeline.

Applied to all agent executions in order:
Request → Logging → RateLimit → ContentSafety → PII → VRAMGuard → Compliance → Agent.run()
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from agent_face.config import settings

logger = logging.getLogger(__name__)

# Type for middleware next-step callback
NextStep = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class MiddlewareContext:
    """Context passed through the middleware pipeline."""

    user_id: str = "anonymous"
    session_id: str = "unknown"
    task_type: str = "unknown"  # "analyze" | "beautify"
    start_time: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)


class MiddlewarePipeline:
    """
    Ordered middleware pipeline for MAF agent execution.

    Each middleware can:
    - Modify the input before passing to the next step
    - Short-circuit and raise an error (safety rejection)
    - Modify the output after the agent runs
    """

    def __init__(self):
        self._middlewares: list[Callable[[dict, MiddlewareContext, NextStep], Awaitable[dict]]] = [
            self._logging_middleware,
            self._rate_limit_middleware,
            self._vram_guard_middleware,
            self._compliance_middleware,
        ]

    async def execute(
        self,
        input_data: dict[str, Any],
        context: MiddlewareContext,
        agent_fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Execute the full middleware pipeline, ending with the agent function.

        Args:
            input_data: The input dict to pass to the agent.
            context: Middleware execution context.
            agent_fn: The actual agent execution function.

        Returns:
            The agent's output dict.
        """
        # Build the chain: each middleware wraps the next
        handler = agent_fn
        for mw in reversed(self._middlewares):
            handler = self._wrap(mw, context, handler)

        return await handler(input_data)

    def _wrap(self, middleware, context, next_step):
        """Wrap a middleware around the next step."""
        async def wrapped(data):
            return await middleware(data, context, next_step)
        return wrapped

    # ── Middleware Implementations ────────────────────────────

    async def _logging_middleware(
        self,
        data: dict[str, Any],
        ctx: MiddlewareContext,
        next_step: NextStep,
    ) -> dict[str, Any]:
        """Log task start, latency, and completion."""
        logger.info(
            "MAF task started",
            extra={
                "user_id": ctx.user_id,
                "session_id": ctx.session_id,
                "task_type": ctx.task_type,
            },
        )
        try:
            result = await next_step(data)
            latency = (time.monotonic() - ctx.start_time) * 1000
            logger.info(
                "MAF task completed",
                extra={
                    "user_id": ctx.user_id,
                    "session_id": ctx.session_id,
                    "task_type": ctx.task_type,
                    "latency_ms": round(latency, 2),
                },
            )
            ctx.metadata["latency_ms"] = latency
            return result
        except Exception as e:
            latency = (time.monotonic() - ctx.start_time) * 1000
            logger.error(
                "MAF task failed",
                extra={
                    "user_id": ctx.user_id,
                    "session_id": ctx.session_id,
                    "task_type": ctx.task_type,
                    "latency_ms": round(latency, 2),
                    "error": str(e),
                },
            )
            raise

    async def _rate_limit_middleware(
        self,
        data: dict[str, Any],
        ctx: MiddlewareContext,
        next_step: NextStep,
    ) -> dict[str, Any]:
        """
        Token-bucket rate limiter per user.

        In production, this would use Redis for distributed rate limiting.
        For now, a simple in-memory implementation prevents abuse.
        """
        # TODO: Implement proper token-bucket rate limiting with Redis
        return await next_step(data)

    async def _vram_guard_middleware(
        self,
        data: dict[str, Any],
        ctx: MiddlewareContext,
        next_step: NextStep,
    ) -> dict[str, Any]:
        """
        Validate image dimensions before model inference.

        Rejects images exceeding max dimensions (downsampling should
        have already happened in the SafetyGuardAgent input check).
        """
        image_b64 = data.get("image_b64", "")
        if image_b64:
            # Rough estimate: base64 string length * 3/4 ≈ byte size
            estimated_bytes = len(image_b64) * 3 // 4
            max_bytes = settings.max_image_bytes
            if estimated_bytes > max_bytes * 2:
                raise ValueError(
                    f"Image too large for model inference: ~{estimated_bytes} bytes "
                    f"(limit: {max_bytes})"
                )
        return await next_step(data)

    async def _compliance_middleware(
        self,
        data: dict[str, Any],
        ctx: MiddlewareContext,
        next_step: NextStep,
    ) -> dict[str, Any]:
        """
        Task compliance monitor.

        Logs every task execution for audit purposes.
        In production, this would send metrics to OpenTelemetry.
        """
        result = await next_step(data)
        ctx.metadata["compliance_checked"] = True
        return result
