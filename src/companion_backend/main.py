from __future__ import annotations

import os

from aiohttp import web

from .app import create_app


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    web.run_app(create_app(), host="0.0.0.0", port=port)
