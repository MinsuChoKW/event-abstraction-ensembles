# src/eae/matching/dp_runner.py

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from eae.matching.segment_match import build_edges_forward_for_case
from eae.matching.bidirectional import (
    build_edges_backward_for_case,
    merge_forward_backward_edges,
)


def write_jsonl_gz(
    rows: Iterable[Dict[str, Any]],
    path: str | Path,
) -> Path:
    """
    Write rows as gzipped JSONL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return path


def build_segment_matches_for_case(
    case: Dict[str, Any],
    pattern_pool: Dict[str, List[List[str]]],
    *,
    jump: int,
    use_backward_matching: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build segment matches for one case.
    """
    case_id = case["case_id"]
    sequence = case["sequence"]

    forward_edges, forward_summary = build_edges_forward_for_case(
        case_id=case_id,
        sequence=sequence,
        pattern_pool=pattern_pool,
        jump=jump,
        direction="forward",
    )

    backward_edges = []
    backward_summary = None

    if use_backward_matching:
        backward_edges, backward_summary = build_edges_backward_for_case(
            case_id=case_id,
            sequence=sequence,
            pattern_pool=pattern_pool,
            jump=jump,
        )

        edges = merge_forward_backward_edges(
            forward_edges,
            backward_edges,
        )
    else:
        edges = forward_edges

    label_counts = Counter(edge["label"] for edge in edges)

    summary = {
        "case_id": str(case_id),
        "case_len": int(len(sequence)),
        "jump": int(jump),
        "use_backward_matching": bool(use_backward_matching),
        "n_edges_forward": int(len(forward_edges)),
        "n_edges_backward": int(len(backward_edges)),
        "n_edges_merged": int(len(edges)),
        "forward_reachable_end": bool(forward_summary["reachable_end"]),
        "backward_reachable_end": bool(backward_summary["reachable_end"])
        if backward_summary is not None
        else None,
        "merged_label_counts": dict(label_counts),
    }

    return edges, summary


def build_segment_matches_for_cases(
    cases: List[Dict[str, Any]],
    pattern_pool: Dict[str, List[List[str]]],
    *,
    jump: int,
    use_backward_matching: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Build segment matches for all cases.

    Returns
    -------
    all_edges
    case_summaries
    edge_label_counts
    """
    all_edges: List[Dict[str, Any]] = []
    case_summaries: List[Dict[str, Any]] = []
    edge_label_counts = Counter()

    for idx, case in enumerate(cases, start=1):
        edges, summary = build_segment_matches_for_case(
            case,
            pattern_pool,
            jump=jump,
            use_backward_matching=use_backward_matching,
        )

        all_edges.extend(edges)
        case_summaries.append(summary)

        for edge in edges:
            edge_label_counts[str(edge["label"])] += 1

        if idx % 100 == 0:
            print(
                f"[Matching] processed {idx}/{len(cases)} cases, "
                f"edges_so_far={len(all_edges)}"
            )

    return all_edges, case_summaries, dict(edge_label_counts)