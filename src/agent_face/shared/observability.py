"""
OpenTelemetry observability setup.

Configures tracing and metrics for the AgentFace service.
In production, traces are exported to an OTLP collector.
In development, tracing is disabled by default.
"""

import logging
from agent_face.config import settings


def setup_observability() -> None:
    """
    Initialize OpenTelemetry tracing if enabled.

    Call this during application startup.
    """
    if not settings.enable_tracing:
        logging.info("OpenTelemetry tracing is disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        provider = TracerProvider()

        if settings.otel_exporter_otlp_endpoint:
            exporter = OTLPSpanExporter(
                endpoint=settings.otel_exporter_otlp_endpoint,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        logging.info(
            f"OpenTelemetry tracing enabled "
            f"(exporter: {settings.otel_exporter_otlp_endpoint or 'console'})"
        )
    except ImportError:
        logging.warning(
            "OpenTelemetry packages not installed. Tracing unavailable."
        )
    except Exception as e:
        logging.error(f"Failed to initialize OpenTelemetry: {e}")


def instrument_fastapi(app) -> None:
    """Instrument a FastAPI app with OpenTelemetry."""
    if not settings.enable_tracing:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass
