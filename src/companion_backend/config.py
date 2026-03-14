from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    app_env: str
    database_url: str
    public_base_url: str | None
    summary_delay_seconds: float
    location_response_cooldown_seconds: float
    gemini_api_key: str | None
    gemini_model: str
    google_places_api_key: str | None
    ticketmaster_api_key: str | None
    gemini_base_url: str
    google_places_base_url: str
    ticketmaster_base_url: str
    google_places_language_code: str
    gemini_live_model: str
    gemini_live_ws_url: str
    gemini_live_audio_idle_seconds: float
    ticketmaster_event_cache_seconds: float

    @classmethod
    def from_env(cls) -> "AppConfig":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            database_path = Path.cwd() / "data" / "companion.db"
            database_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite+pysqlite:///{database_path}"

        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            database_url=database_url,
            public_base_url=os.getenv("PUBLIC_BASE_URL"),
            summary_delay_seconds=float(os.getenv("SUMMARY_DELAY_SECONDS", "0.05")),
            location_response_cooldown_seconds=float(
                os.getenv("LOCATION_RESPONSE_COOLDOWN_SECONDS", "15")
            ),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_places_api_key=os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY"),
            ticketmaster_api_key=os.getenv("TICKETMASTER_API_KEY"),
            gemini_base_url=os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com",
            ),
            google_places_base_url=os.getenv(
                "GOOGLE_PLACES_BASE_URL",
                "https://places.googleapis.com",
            ),
            ticketmaster_base_url=os.getenv(
                "TICKETMASTER_BASE_URL",
                "https://app.ticketmaster.com/discovery/v2",
            ),
            google_places_language_code=os.getenv("GOOGLE_PLACES_LANGUAGE_CODE", "en"),
            gemini_live_model=os.getenv(
                "GEMINI_LIVE_MODEL",
                "gemini-2.5-flash-native-audio-preview-12-2025",
            ),
            gemini_live_ws_url=os.getenv(
                "GEMINI_LIVE_WS_URL",
                "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
            ),
            gemini_live_audio_idle_seconds=float(
                os.getenv("GEMINI_LIVE_AUDIO_IDLE_SECONDS", "2.0")
            ),
            ticketmaster_event_cache_seconds=float(
                os.getenv("TICKETMASTER_EVENT_CACHE_SECONDS", "180")
            ),
        )
