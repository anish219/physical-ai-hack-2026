"""API keys from a local env file (KEY=value lines). Values are never logged, printed or returned in errors."""
from __future__ import annotations

import os
import pathlib

from hillclimb.common.config import CFG

_cache: dict[str, str] | None = None


def _read(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in pathlib.Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001  a missing file means no keys, reported as unhealthy by the callers
        pass
    return out


def get(name: str) -> str | None:
    """The key from the environment first, then the keys file. None when absent."""
    global _cache
    if os.environ.get(name):
        return os.environ[name]
    if _cache is None:
        _cache = _read(CFG["KEYS_FILE"])
    return _cache.get(name) or None


def present(name: str) -> bool:
    return bool(get(name))
