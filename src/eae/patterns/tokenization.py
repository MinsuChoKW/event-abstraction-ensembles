# src/eae/patterns/tokenization.py

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from eae.paths import build_case_strings_path


def normalize_sequence(value: Any) -> List[str]:
    """
    Normalize sequence-like values into List[str].

    Supported:
      ["a", "b", "c"]
      "a b c"
      "abc"
      '["a", "b", "c"]'
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    if isinstance(value, tuple):
        return [str(x) for x in value]

    if isinstance(value, str):
        s = value.strip()

        if not s:
            return []

        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass

        if " " in s:
            return [x for x in s.split() if x]

        return list(s)

    raise ValueError(f"Unsupported sequence value type: {type(value)}")


def read_jsonl_gz(path: str | Path) -> Iterable[Dict[str, Any]]:
    """
    Read gzipped JSONL file.
    """
    path = Path(path)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_case_strings(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load case strings from dataset-specific processed directory.

    Accepted record formats:
      {"case_id": "...", "sequence": [...]}
      {"case_id": "...", "case_string": "abc"}
      {"case_id": "...", "trace": "a b c"}
    """
    path = build_case_strings_path(cfg)

    if not path.exists():
        raise FileNotFoundError(
            f"Case strings file does not exist: {path}\n"
            "Expected dataset-specific path such as:\n"
            "  data/processed/{dataset.name}/case_strings_N{n_cases}.jsonl.gz"
        )

    cases = []

    for idx, row in enumerate(read_jsonl_gz(path)):
        case_id = (
            row.get("case_id")
            or row.get("case:concept:name")
            or row.get("id")
            or f"case_{idx}"
        )

        seq_value = (
            row.get("sequence")
            or row.get("events")
            or row.get("activities")
            or row.get("case_string")
            or row.get("trace")
            or row.get("seq")
        )

        seq = normalize_sequence(seq_value)

        cases.append(
            {
                "case_id": str(case_id),
                "sequence": seq,
                "length": len(seq),
            }
        )

    return cases