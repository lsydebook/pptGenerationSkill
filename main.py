"""Application entrypoint: uv run main.py"""

from __future__ import annotations

import os

import uvicorn

from app.api.fastapi_app import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() in {"1", "true", "yes"},
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
