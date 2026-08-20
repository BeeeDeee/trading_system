"""FastAPI application factory."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the external control-plane application."""
    return FastAPI(title="AI Trading Platform", version="0.1.0")


app = create_app()
