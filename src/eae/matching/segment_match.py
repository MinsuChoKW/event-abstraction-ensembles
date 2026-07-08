# src/eae/matching/segment_match.py

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from eae.matching.alignment import alignment_cost


def flatten_pattern_pool(
    pattern_pool: Dict[str, List[List[str]]],
) -> List[Dict[str, Any]]:
    """
    Convert label -> paths into flat pattern records.
    """
    rows = []

    for label in sorted(pattern_pool.keys()):
        for path_idx, path in enumerate(pattern_pool[label]):
            rows.append(
                {
                    "label": str(label),
                    "path_idx": int(path_idx),
                    "path": [str(x) for x in path],
                    "path_len": int(len(path)),
                }
            )

    return rows


def index_patterns_by_length(
    flat_patterns: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Index flattened patterns by path length.
    """
    by_len = defaultdict(list)

    for item in flat_patterns:
        by_len[int(item["path_len"])].append(item)

    return dict(by_len)


def build_edges_forward_for_case(
    *,
    case_id: str,
    sequence: List[str],
    pattern_pool: Dict[str, List[List[str]]],
    jump: int,
    direction: str = "forward",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Forward segment-match construction.

    For path length L and terminal position j:
      i ranges from j - L - jump to j - L

    Therefore the matched case segment length ranges:
      L <= len(segment) <= L + jump
    """
    T = len(sequence)

    flat_patterns = flatten_pattern_pool(pattern_pool)
    by_len = index_patterns_by_length(flat_patterns)

    reachable = [False] * (T + 1)
    reachable[0] = True

    edges: List[Dict[str, Any]] = []
    label_counts = Counter()

    for j in range(1, T + 1):
        for L, patterns in by_len.items():
            i_min = max(0, j - L - int(jump))
            i_max = min(j - 1, j - L)

            if i_min > i_max:
                continue

            for i in range(i_min, i_max + 1):
                if not reachable[i]:
                    continue

                segment = sequence[i:j]

                for pat in patterns:
                    cost = alignment_cost(pat["path"], segment)

                    edge = {
                        "case_id": str(case_id),
                        "i": int(i),
                        "j": int(j),
                        "label": str(pat["label"]),
                        "path_idx": int(pat["path_idx"]),
                        "path": pat["path"],
                        "path_len": int(pat["path_len"]),
                        "segment": list(segment),
                        "segment_len": int(len(segment)),
                        "cost": int(cost),
                        "direction": direction,
                    }

                    edges.append(edge)
                    label_counts[str(pat["label"])] += 1
                    reachable[j] = True

    summary = {
        "case_id": str(case_id),
        "case_len": int(T),
        "direction": direction,
        "n_edges": int(len(edges)),
        "reachable_end": bool(reachable[T]),
        "n_reachable_positions": int(sum(reachable)),
        "edge_counts_by_label": dict(label_counts),
    }

    return edges, summary