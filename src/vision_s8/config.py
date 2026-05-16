"""
Configuration management for Vision_S8.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    gemini_api_key: str = Field(default="", description="Google Gemini API Key")
    
    # Application API Key (for securing your Vision_S8 API)
    api_key: str | None = Field(
        default=None, 
        description="Optional API key to protect Vision_S8 endpoints. If set, clients must provide X-API-Key header."
    )
    api_key_enabled: bool = Field(
        default=False,
        description="Enable API key authentication. Recommended for production."
    )

    # Server Settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # CORS Settings
    cors_origins: list[str] = Field(
        default=["http://localhost", "http://localhost:3000", "http://127.0.0.1"],
        description="Allowed CORS origins"
    )

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./vision_s8.db",
        description="Database connection URL",
    )

    # File Storage
    upload_dir: Path = Field(default=Path("./uploads"), description="Upload directory")
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    max_upload_size_mb: int = Field(default=50, ge=1, le=500, description="Max upload size in MB")

    # AI Model Settings
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model to use")
    max_retries: int = Field(default=3, ge=1, le=10, description="Max API retries")
    retry_delay: float = Field(default=1.0, ge=0.1, le=30.0, description="Retry delay in seconds")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_file: Path | None = Field(default=None, description="Log file path")

    # Rate Limiting
    rate_limit_rpm: int = Field(default=60, ge=1, le=1000, description="Rate limit per minute")

    # Marketplace APIs (Optional)
    amazon_access_key: str | None = Field(default=None, description="Amazon PAAPI access key")
    amazon_secret_key: str | None = Field(default=None, description="Amazon PAAPI secret key")
    amazon_partner_tag: str | None = Field(default=None, description="Amazon partner tag")
    ebay_app_id: str | None = Field(default=None, description="eBay API app ID")

    @field_validator("upload_dir", "output_dir", mode="after")
    @classmethod
    def ensure_directory_exists(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def max_upload_size_bytes(self) -> int:
        """Return max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def base_dir(self) -> Path:
        """Return base directory of the project."""
        return Path(__file__).parent.parent.parent.parent

    @property
    def prompts_dir(self) -> Path:
        """Return prompts directory."""
        return Path(__file__).parent / "data" / "prompts"

    @property
    def platform_rules_path(self) -> Path:
        """Return platform rules JSON path."""
        return Path(__file__).parent / "data" / "platform_rules.json"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for quick access
settings = get_settings()
