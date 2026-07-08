# src/eae/selection/path_solution.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class PathSolution:
    """
    One decomposition / combination candidate for a case.

    Ranking principle
    -----------------
    1. Maximize coverage.
    2. Minimize total alignment cost.
    3. Minimize number of segments.
    4. Lexicographic order of label sequence.
    5. Lexicographic order of path sequence.
    """
    case_id: str
    case_len: int
    terminal_pos: int
    coverage: int
    total_cost: int
    edges: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def n_segments(self) -> int:
        return len(self.edges)

    @property
    def coverage_ratio(self) -> float:
        if self.case_len <= 0:
            return 0.0
        return float(self.coverage) / float(self.case_len)

    @property
    def label_sequence(self) -> Tuple[str, ...]:
        return tuple(str(e["label"]) for e in self.edges)

    @property
    def path_sequence_key(self) -> Tuple[str, ...]:
        return tuple(
            " ".join(str(x) for x in e.get("path", []))
            for e in self.edges
        )

    @property
    def span_sequence_key(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (int(e["i"]), int(e["j"]))
            for e in self.edges
        )

    def rank_key(self) -> Tuple[Any, ...]:
        """
        Lower rank_key is better.

        coverage is negated because we want larger coverage first.
        """
        return (
            -int(self.coverage),
            int(self.total_cost),
            int(self.n_segments),
            self.label_sequence,
            self.path_sequence_key,
            self.span_sequence_key,
        )

    def dedup_key(self) -> Tuple[Any, ...]:
        """
        Key for removing duplicate solutions.
        """
        return tuple(
            (
                int(e["i"]),
                int(e["j"]),
                str(e["label"]),
                tuple(str(x) for x in e.get("path", [])),
                int(e["cost"]),
            )
            for e in self.edges
        )

    def to_dict(self, *, rank: int | None = None) -> Dict[str, Any]:
        """
        Convert to serializable dictionary.
        """
        return {
            "rank": rank,
            "case_id": self.case_id,
            "case_len": int(self.case_len),
            "terminal_pos": int(self.terminal_pos),
            "coverage": int(self.coverage),
            "coverage_ratio": self.coverage_ratio,
            "total_cost": int(self.total_cost),
            "n_segments": int(self.n_segments),
            "label_sequence": list(self.label_sequence),
            "edges": [dict(e) for e in self.edges],
        }


def make_empty_solution(
    *,
    case_id: str,
    case_len: int,
) -> PathSolution:
    return PathSolution(
        case_id=str(case_id),
        case_len=int(case_len),
        terminal_pos=0,
        coverage=0,
        total_cost=0,
        edges=tuple(),
    )


def extend_solution(
    sol: PathSolution,
    edge: Dict[str, Any],
) -> PathSolution:
    """
    Extend a solution with one edge.
    """
    i = int(edge["i"])
    j = int(edge["j"])

    if i != sol.terminal_pos:
        raise ValueError(
            f"Edge does not continue solution: "
            f"terminal={sol.terminal_pos}, edge_i={i}, edge_j={j}"
        )

    return PathSolution(
        case_id=sol.case_id,
        case_len=sol.case_len,
        terminal_pos=j,
        coverage=int(sol.coverage) + int(j - i),
        total_cost=int(sol.total_cost) + int(edge["cost"]),
        edges=sol.edges + (dict(edge),),
    )


def deduplicate_solutions(
    solutions: List[PathSolution],
) -> List[PathSolution]:
    """
    Remove duplicate solutions while preserving rank order.
    """
    seen = set()
    out = []

    for sol in sorted(solutions, key=lambda x: x.rank_key()):
        key = sol.dedup_key()

        if key in seen:
            continue

        seen.add(key)
        out.append(sol)

    return out


def keep_top_solutions(
    solutions: List[PathSolution],
    *,
    top_k: int,
) -> List[PathSolution]:
    """
    Keep top-k ranked unique solutions.
    """
    if top_k <= 0:
        return []

    unique = deduplicate_solutions(solutions)
    return sorted(unique, key=lambda x: x.rank_key())[:top_k]