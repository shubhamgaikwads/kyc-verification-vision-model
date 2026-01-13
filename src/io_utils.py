import json
import re
from pathlib import Path
from typing import Any, Dict


def safe_stem(filename: str) -> str:
    """
    Make a safe filename stem for outputs (keeps readability).
    """
    stem = Path(filename).stem
    stem = stem.strip()
    stem = re.sub(r"\s+", "_", stem)         # spaces -> _
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "", stem)  # remove odd chars
    return stem or "document"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
