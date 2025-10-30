import os
import json
from typing import Any, Dict


def save_record_jsonl(record: Dict[str, Any], path: str) -> None:
    """Append a single JSON record to a JSONL file (atomic per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_completed_indices(path: str) -> int:
    """Return last image_id found in a JSONL file, or -1 if none.

    Mirrors previous behavior where the function tracked the last processed
    id and printed it for visibility.
    """
    last_id = -1
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    last_id = record.get("image_id", last_id)
                except Exception:
                    continue
    print(last_id)
    return last_id

