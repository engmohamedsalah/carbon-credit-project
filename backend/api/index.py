"""Vercel serverless entrypoint.

Vercel's Python runtime auto-detects the ASGI ``app``. We add the backend root to
sys.path so ``main`` and its siblings (config, dbdriver, services, utils) import.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402  (exposes the FastAPI app at /api/v1/*)
