"""I/O helpers for experiment outputs."""

from __future__ import annotations

import json
import os
from typing import Any


def save_json(path: str, payload: dict[str, Any]) -> None:
    """Save a JSON file with pretty formatting."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    save_json("results/_io_sanity.json", {"ok": True})
    print("Wrote results/_io_sanity.json")

