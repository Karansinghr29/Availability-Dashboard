"""
config.py
=========

Loads temporary business overrides from ``Availability_AI/config/``.

The only current override is a list of apartment codes treated as Inactive
until the database provides a real ``apartment_status`` / lifecycle column.

When that column exists:
  1. Empty ``inactive_apartment_codes`` in the JSON (or delete the file).
  2. ``annotate_lifecycle_status`` will use DB-derived status only.
  3. Recommendation / search / occupancy keep calling the same APIs —
     no engine changes required.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet

# Availability_AI/config/  (sibling of src/)
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_INACTIVE_APARTMENTS_PATH = _CONFIG_DIR / "inactive_apartments.json"


@lru_cache(maxsize=1)
def get_temporarily_inactive_apartments() -> FrozenSet[str]:
    """Apartment codes forced to Inactive while lifecycle fields are missing.

    Reads ``config/inactive_apartments.json``. Missing/empty file → empty set
    (no temporary overrides). Codes are upper-cased for stable matching.
    """
    path = _INACTIVE_APARTMENTS_PATH
    if not path.is_file():
        return frozenset()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()

    raw = payload.get("inactive_apartment_codes", []) if isinstance(payload, dict) else []
    if not isinstance(raw, list):
        return frozenset()
    codes = {
        str(code).strip().upper()
        for code in raw
        if code is not None and str(code).strip()
    }
    return frozenset(codes)


def clear_config_cache() -> None:
    """Clear cached config (tests / hot-reload after editing the JSON)."""
    get_temporarily_inactive_apartments.cache_clear()
