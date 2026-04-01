"""ASGI entry: `uvicorn ai_ops_chat.main:app --host 0.0.0.0 --port 8000`."""

from ai_ops_chat.api import app

__all__ = ["app"]
