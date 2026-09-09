"""PRAMAAN - proof-grounded eligibility guidance for Indian welfare schemes."""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


def _load_dotenv() -> None:
    """Read .env into the environment, if it exists.

    Done by hand rather than with python-dotenv: it is a dozen lines, and every
    dependency dropped is one less thing that can fail on a free host. Existing
    environment variables always win, so a real deployment's secrets are never
    overwritten by a stray .env left in the working tree.

    This lives in __init__ so it runs before config.py reads os.getenv at
    import time -- which is the whole reason a .env is useful.
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()
