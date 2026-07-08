# src/eae/patterns/pool_builder.py

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from eae.paths import build_lpm_path_file, build_session_topk_file


def load_json(path: str | Path) -> Any:
    """
    Load JSON file.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int = 2,
) -> Path:
    """
    Save JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

    return path


def save_json_gz(
    path: str | Path,
    obj: Any,
) -> Path:
    """
    Save gzipped JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

    return path


def load_lpm_pattern_pool(
    cfg: Dict[str, Any],
) -> Dict[str, List[List[str]]]:
    """
    Load LPM path file and map each LPM group to G001, G002, ...

    Input path is resolved by:
      eae.paths.build_lpm_path_file(cfg)

    Example:
      data/bpic2015/interim/LPM_Path/lpm_group_paths_loop2_remove_dup.json
    """
    label_prefix = cfg["abstraction_source"]["label_prefix"]["lpm"]
    lpm_file = build_lpm_path_file(cfg)

    if not lpm_file.exists():
        raise FileNotFoundError(f"LPM path file does not exist: {lpm_file}")

    payload = load_json(lpm_file)
    groups = payload.get("groups", {})

    pool: Dict[str, List[List[str]]] = defaultdict(list)

    for group_rank, gkey in enumerate(sorted(groups.keys()), start=1):
        label = f"{label_prefix}{group_rank:03d}"

        group_payload = groups[gkey]

        for _, item in group_payload.items():
            for path in item.get("paths", []):
                path = [str(x) for x in path]

                if path:
                    pool[label].append(path)

    return dict(pool)


def load_session_pattern_pool(
    cfg: Dict[str, Any],
    *,
    k: int,
) -> Dict[str, List[List[str]]]:
    """
    Load session Top-K pattern file.

    Input path is resolved by:
      eae.paths.build_session_topk_file(cfg)

    Example:
      data/bpic2015/interim/cut_exports/session_dbscan_cut_topK_K060.json
    """
    session_file = build_session_topk_file(cfg, k=k)

    if not session_file.exists():
        raise FileNotFoundError(f"Session top-K file does not exist: {session_file}")

    payload = load_json(session_file)
    patterns = payload.get("patterns", [])

    pool: Dict[str, List[List[str]]] = defaultdict(list)

    for item in patterns:
        label = str(item["label"])
        seq = item.get("sequence", [])

        seq = [str(x) for x in seq]

        if seq:
            pool[label].append(seq)

    return dict(pool)


def merge_pattern_pools(
    pools: List[Dict[str, List[List[str]]]],
) -> Dict[str, List[List[str]]]:
    """
    Merge multiple pattern pools.
    """
    merged: Dict[str, List[List[str]]] = defaultdict(list)

    for pool in pools:
        for label, paths in pool.items():
            merged[label].extend(paths)

    return dict(merged)


def deduplicate_paths_within_label(
    pool: Dict[str, List[List[str]]],
) -> Dict[str, List[List[str]]]:
    """
    Remove duplicate paths within each label.
    """
    out: Dict[str, List[List[str]]] = {}

    for label, paths in pool.items():
        seen = set()
        kept = []

        for path in paths:
            key = tuple(str(x) for x in path)

            if key in seen:
                continue

            seen.add(key)
            kept.append(list(key))

        out[label] = kept

    return out


def build_pattern_pool(
    cfg: Dict[str, Any],
    *,
    k: int,
) -> Dict[str, List[List[str]]]:
    """
    Build unified pattern pool according to abstraction_source.method.

    method:
      BOTH
      LPM
      SESSION
    """
    method = cfg["abstraction_source"]["method"].upper()

    pools = []

    if method in {"BOTH", "LPM"}:
        pools.append(load_lpm_pattern_pool(cfg))

    if method in {"BOTH", "SESSION"}:
        pools.append(load_session_pattern_pool(cfg, k=k))

    if not pools:
        raise ValueError(f"Invalid abstraction_source.method: {method}")

    pool = merge_pattern_pools(pools)
    pool = deduplicate_paths_within_label(pool)

    return pool


def summarize_pattern_pool(
    pool: Dict[str, List[List[str]]],
) -> Dict[str, Any]:
    """
    Summarize pattern pool.
    """
    length_counts = defaultdict(int)
    per_label = {}

    total_paths = 0

    for label, paths in sorted(pool.items()):
        lengths = [len(path) for path in paths]

        for length in lengths:
            length_counts[int(length)] += 1

        per_label[label] = {
            "n_paths": int(len(paths)),
            "min_len": int(min(lengths)) if lengths else None,
            "max_len": int(max(lengths)) if lengths else None,
            "lengths": sorted(int(x) for x in set(lengths)),
        }

        total_paths += len(paths)

    return {
        "n_labels": int(len(pool)),
        "n_paths": int(total_paths),
        "path_length_counts": dict(sorted(length_counts.items())),
        "per_label": per_label,
    }


def save_pattern_pool_artifacts(
    pool: Dict[str, List[List[str]]],
    *,
    cfg: Dict[str, Any],
    k: int,
    jump: int,
    output_paths: Dict[str, Path],
) -> Dict[str, Any]:
    """
    Save pattern pool artifacts under run folder.

    Outputs:
      pattern_pool/pattern_pool_by_label.json.gz
      pattern_pool/pattern_pool_meta.json
    """
    method = cfg["abstraction_source"]["method"].upper()

    meta = {
        "dataset": cfg["dataset"]["name"],
        "method": method,
        "k": int(k),
        "jump": int(jump),
        "summary": summarize_pattern_pool(pool),
        "lpm_path_file": str(build_lpm_path_file(cfg))
        if method in {"BOTH", "LPM"}
        else None,
        "session_topk_file": str(build_session_topk_file(cfg, k=k))
        if method in {"BOTH", "SESSION"}
        else None,
    }

    save_json_gz(
        output_paths["pattern_pool_by_label"],
        pool,
    )

    save_json(
        output_paths["pattern_pool_meta"],
        meta,
    )

    return meta