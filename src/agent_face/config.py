"""
Application configuration via pydantic-settings.

All settings are loaded from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """AgentFace application settings."""

    # ── Database ──
    database_url: str = "sqlite:///./data/agent_face.db"

    # ── Model Services ──
    multimodal_model_url: str = "http://localhost:8001"
    beautify_model_url: str = "http://localhost:8899"
    model_request_timeout: float = 180.0

    # ── Beautification Model ──
    beauty_model_name: str = "deepfrr"     # deepfrr or ffhqr
    beauty_steps: int = 30                 # inference steps (higher = better, slower)
    beauty_guidance_scale: float = 3.0     # text guidance strength
    beauty_image_guidance_scale: float = 1.5  # original image preservation
    beauty_seed: int = 42

    # ── MIMO API ──
    mimo_api_key: str = ""
    mimo_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    mimo_model: str = "mimo-v2-omni"

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # ── Image Constraints ──
    max_image_dimension: int = 4096
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_image_formats: str = "image/jpeg,image/png,image/webp"

    # ── Memory ──
    preference_learning_rate: float = 0.4  # EMA 学习率，越高越"听话"
    min_sessions_for_automation: int = 1  # 从第一次就开始学习

    # ── Safety ──
    enable_content_safety: bool = True
    enable_pii_detection: bool = False
    over_smoothing_threshold: float = 0.15

    # ── Observability ──
    otel_exporter_otlp_endpoint: Optional[str] = None
    enable_tracing: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Allow MIMO_* and other extra env vars
    }

    @property
    def allowed_formats_list(self) -> list[str]:
        """Parse the comma-separated allowed formats into a list."""
        return [f.strip() for f in self.allowed_image_formats.split(",")]


# Global settings instance
settings = Settings()
