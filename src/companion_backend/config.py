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
        )
