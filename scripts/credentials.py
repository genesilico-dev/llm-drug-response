"""Read the API credential locally; never print it or include it in artifacts."""

import os
from pathlib import Path

from dotenv import dotenv_values


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        config = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
        key = config.get("OPENROUTER_API_KEY") or config.get("OPENROUTER")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY locally before running annotations")
    return key
