"""Thin read-only HTTP API over published ``current/`` artifacts (#106)."""

from services.api.app import app, create_app

__all__ = ["app", "create_app"]
