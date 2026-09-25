"""Shared pytest fixtures. Force a non-interactive matplotlib backend."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
