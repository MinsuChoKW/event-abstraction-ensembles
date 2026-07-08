# src/eae/matching/bidirectional.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from eae.matching.segment_match import build_edges_forward_for_case


def reverse_pattern_pool(
    pattern_pool: Dict[str, List[List[str]]],
) -> Dict[str, List[List[str]]]:
    """
    Reverse every path in pattern pool.
    """
    return {
        label: [list(reversed(path)) for path in paths]
        for label, paths in pattern_pool.items()
    }


def remap_reversed_edge_to_original(
    edge: Dict[str, Any],
    *,
    case_len: int,
) -> Dict[str, Any]:
    """
    Convert reversed-sequence edge coordinates back to original coordinates.

    reversed:
      i_rev -> j_rev

    original:
      case_len - j_rev -> case_len - i_rev
    """
    i_rev = int(edge["i"])
    j_rev = int(edge["j"])

    out = dict(edge)
    out["i"] = int(case_len - j_rev)
    out["j"] = int(case_len - i_rev)
    out["segment"] = list(reversed(edge["segment"]))
    out["path"] = list(reversed(edge["path"]))
    out["direction"] = "backward"

    return out


def build_edges_backward_for_case(
    *,
    case_id: str,
    sequence: List[str],
    pattern_pool: Dict[str, List[List[str]]],
    jump: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Backward matching via reversed sequence and reversed pattern pool.
    """
    reversed_sequence = list(reversed(sequence))
    reversed_pool = reverse_pattern_pool(pattern_pool)

    rev_edges, rev_summary = build_edges_forward_for_case(
        case_id=case_id,
        sequence=reversed_sequence,
        pattern_pool=reversed_pool,
        jump=jump,
        direction="backward_reversed",
    )

    T = len(sequence)

    edges = [
        remap_reversed_edge_to_original(edge, case_len=T)
        for edge in rev_edges
    ]

    label_counts = Counter(edge["label"] for edge in edges)

    summary = {
        "case_id": str(case_id),
        "case_len": int(T),
        "direction": "backward",
        "n_edges": int(len(edges)),
        "reachable_end": bool(rev_summary["reachable_end"]),
        "n_reachable_positions": int(rev_summary["n_reachable_positions"]),
        "edge_counts_by_label": dict(label_counts),
    }

    return edges, summary


def edge_key(edge: Dict[str, Any]) -> tuple:
    """
    Key for duplicate detection between forward/backward edges.
    """
    return (
        edge["case_id"],
        int(edge["i"]),
        int(edge["j"]),
        str(edge["label"]),
        tuple(edge["path"]),
        int(edge["cost"]),
    )


def merge_forward_backward_edges(
    forward_edges: List[Dict[str, Any]],
    backward_edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge forward and backward edges.

    If identical edge appears in both:
      direction = both
    """
    merged = {}

    for edge in forward_edges:
        item = dict(edge)
        item["direction"] = "forward"
        merged[edge_key(item)] = item

    for edge in backward_edges:
        k = edge_key(edge)

        if k in merged:
            merged[k]["direction"] = "both"
        else:
            item = dict(edge)
            item["direction"] = "backward"
            merged[k] = item

    return list(merged.values())