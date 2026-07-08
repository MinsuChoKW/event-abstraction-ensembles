# src/eae/selection/topk_solver.py

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MAX_STORED_COMBINATIONS = 5


# ============================================================
# IO
# ============================================================

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


# ============================================================
# Lightweight original-notebook style objects
# ============================================================

@dataclass(frozen=True)
class Edge:
    """
    Original notebook style edge.

    j: start position
    i: end position
    L: label
    D: alignment cost
    seg_len: covered segment length
    """
    j: int
    i: int
    L: str
    D: float
    seg_len: int
    path: Tuple[str, ...] = tuple()
    segment: Tuple[str, ...] = tuple()
    direction: str = ""


@dataclass
class DPResultLite:
    n: int
    dp_edges_by_i: List[List[Edge]]


@dataclass(frozen=True)
class PathSol:
    cost: float
    end_i: int
    edges: Tuple[Edge, ...]


# ============================================================
# Ranking
# ============================================================

def label_seq(path: PathSol) -> Tuple[str, ...]:
    return tuple(e.L for e in path.edges)


def path_seq_key(path: PathSol) -> Tuple[str, ...]:
    return tuple(
        " ".join(e.path)
        for e in path.edges
    )


def keep_top_k_paths(
    paths: List[PathSol],
    k: int,
) -> List[PathSol]:
    """
    Keep top-k paths by the original objective inside fixed coverage:

      1. smaller total cost
      2. smaller number of edges
      3. lexicographic label sequence
      4. lexicographic path sequence
    """
    if k <= 0 or not paths:
        return []

    paths_sorted = sorted(
        paths,
        key=lambda p: (
            float(p.cost),
            len(p.edges),
            label_seq(p),
            path_seq_key(p),
        ),
    )

    return paths_sorted[:k]


# ============================================================
# Original notebook style DP
# ============================================================

def find_topk_partition_paths_to_end(
    res: DPResultLite,
    *,
    end_i: int,
    topk: int,
    rng: random.Random,
) -> List[PathSol]:
    """
    Find top-k connected path combinations that end exactly at end_i.

    This mirrors the original notebook structure:

      dp_paths[0] = empty path
      for i = 1..end_i:
          for incoming edge e ending at i:
              extend paths from e.j
          dp_paths[i] = keep_top_k(candidates, topk)

    Important:
      This keeps only topk paths per prefix position.
      It does NOT keep beam_width=50 states.
    """
    n = res.n

    if end_i < 0 or end_i > n:
        return []

    topk = min(int(topk), MAX_STORED_COMBINATIONS)

    dp_paths: List[List[PathSol]] = [
        []
        for _ in range(end_i + 1)
    ]

    dp_paths[0] = [
        PathSol(
            cost=0.0,
            end_i=0,
            edges=tuple(),
        )
    ]

    for i in range(1, end_i + 1):
        cand: List[PathSol] = []

        for e in res.dp_edges_by_i[i]:
            j = int(e.j)

            if j < 0 or j >= i:
                continue

            if not dp_paths[j]:
                continue

            for prev in dp_paths[j]:
                cand.append(
                    PathSol(
                        cost=float(prev.cost) + float(e.D),
                        end_i=i,
                        edges=prev.edges + (e,),
                    )
                )

        dp_paths[i] = keep_top_k_paths(
            cand,
            topk,
        )

    return dp_paths[end_i]


def find_topk_closest_coverage_paths(
    res: DPResultLite,
    *,
    topk: int = 1,
    max_delta: int = 50,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[int], List[PathSol]]:
    """
    Original notebook style outer coverage priority.

    Try full coverage first:
      delta = 0, end_i = n

    If impossible, try:
      delta = 1, end_i = n - 1
      delta = 2, end_i = n - 2
      ...

    Return first feasible delta and its top-k solutions.
    """
    if rng is None:
        rng = random.Random(0)

    n = int(res.n)
    topk = min(int(topk), MAX_STORED_COMBINATIONS)

    for delta in range(0, int(max_delta) + 1):
        end_i = n - delta

        if end_i < 0:
            continue

        sols = find_topk_partition_paths_to_end(
            res,
            end_i=end_i,
            topk=topk,
            rng=rng,
        )

        if sols:
            return delta, sols

    return None, []


# ============================================================
# Edge conversion
# ============================================================

def normalize_raw_edge_to_original_style(
    rec: Dict[str, Any],
) -> Edge:
    """
    Convert current JSON edge record to original notebook style Edge.

    Current stage-03 edge convention:
      rec["i"] = start position
      rec["j"] = end position

    Original notebook Edge convention:
      Edge.j = start position
      Edge.i = end position

    Therefore:
      start = rec["i"]
      end   = rec["j"]
      Edge(j=start, i=end)
    """
    start = int(rec["i"])
    end = int(rec["j"])

    path = tuple(str(x) for x in rec.get("path", []))
    segment = tuple(str(x) for x in rec.get("segment", []))

    seg_len = int(
        rec.get(
            "seg_len",
            rec.get(
                "segment_len",
                end - start,
            ),
        )
    )

    return Edge(
        j=start,
        i=end,
        L=str(rec.get("label", "")),
        D=float(rec.get("cost", rec.get("D", 0.0))),
        seg_len=seg_len,
        path=path,
        segment=segment,
        direction=str(rec.get("direction", "")),
    )


def load_edges_by_case_original_style(
    edges_path: str | Path,
    *,
    allowed_case_ids: Optional[set[str]] = None,
    progress_every: int = 1_000_000,
) -> Dict[str, Dict[int, List[Edge]]]:
    """
    Load dp_edges.jsonl.gz and group as:

      case_id -> end_position i -> list[Edge]

    This follows the original notebook structure:
      edges_by_case[cid][i].append(Edge(...))
    """
    edges_path = Path(edges_path)

    edges_by_case: Dict[str, Dict[int, List[Edge]]] = defaultdict(
        lambda: defaultdict(list)
    )

    n_edges = 0

    for rec in read_jsonl_gz(edges_path):
        cid = str(rec.get("case_id", ""))

        if allowed_case_ids is not None and cid not in allowed_case_ids:
            continue

        e = normalize_raw_edge_to_original_style(rec)

        edges_by_case[cid][int(e.i)].append(e)

        n_edges += 1

        if progress_every and n_edges % progress_every == 0:
            print(
                f"[Selection] loaded edges: {n_edges:,}",
                flush=True,
            )

    print(
        f"[Selection] finished loading edges: {n_edges:,}",
        flush=True,
    )

    return {
        cid: dict(by_end)
        for cid, by_end in edges_by_case.items()
    }


# ============================================================
# Output conversion
# ============================================================

def edge_to_dict(e: Edge) -> Dict[str, Any]:
    """
    Save edge in current project convention:

      i = start
      j = end
    """
    return {
        "i": int(e.j),
        "j": int(e.i),
        "label": str(e.L),
        "cost": float(e.D),
        "seg_len": int(e.seg_len),
        "segment_len": int(e.seg_len),
        "path": list(e.path),
        "segment": list(e.segment),
        "direction": str(e.direction),
    }


def solution_to_dict(
    sol: PathSol,
    *,
    case_id: str,
    case_len: int,
    rank: int,
    delta: Optional[int],
) -> Dict[str, Any]:
    edges = sol.edges or tuple()

    end_i = int(sol.end_i)
    coverage = end_i
    coverage_ratio = float(coverage / case_len) if case_len else 0.0

    return {
        "case_id": str(case_id),
        "rank": int(rank),
        "case_len": int(case_len),
        "delta": int(delta) if delta is not None else None,
        "terminal_pos": int(end_i),
        "end_i": int(end_i),
        "coverage": int(coverage),
        "coverage_ratio": float(coverage_ratio),
        "total_cost": float(sol.cost),
        "cost": float(sol.cost),
        "n_segments": int(len(edges)),
        "n_edges": int(len(edges)),
        "label_sequence": [str(e.L) for e in edges],
        "path_sequence_key": [
            " ".join(e.path)
            for e in edges
        ],
        "edges": [
            edge_to_dict(e)
            for e in edges
        ],
    }


def build_case_output_row(
    *,
    case_id: str,
    case_len: int,
    n_input_edges: int,
    delta: Optional[int],
    solutions: List[PathSol],
) -> Dict[str, Any]:
    best = solutions[0] if solutions else None

    return {
        "case_id": str(case_id),
        "case_len": int(case_len),
        "n_edges": int(n_input_edges),
        "n_input_edges": int(n_input_edges),
        "n_selected_solutions": int(len(solutions)),
        "n_selected": int(len(solutions)),
        "best_delta": int(delta) if delta is not None else None,
        "best_coverage": int(best.end_i) if best else 0,
        "best_coverage_ratio": float(best.end_i / case_len) if best and case_len else 0.0,
        "best_total_cost": float(best.cost) if best else None,
        "best_n_segments": int(len(best.edges)) if best else None,
        "reaches_end": bool(best.end_i == case_len) if best else False,
        "best_label_sequence": [str(e.L) for e in best.edges] if best else [],
    }


# ============================================================
# Main API for script 04
# ============================================================

def select_top_combinations_for_cases(
    edges_path: str | Path,
    *,
    case_lengths: Dict[str, int],
    top_k: int = 5,
    max_delta: int = 50,
    random_seed: int = 42,
    progress_every: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Select top-k path combinations for every case.

    This version follows the original notebook logic:

      1. load edges into case -> end-position buckets
      2. for each case, build DPResultLite
      3. find_topk_closest_coverage_paths()
      4. save one row per selected solution
    """
    edges_path = Path(edges_path)
    top_k = min(int(top_k), MAX_STORED_COMBINATIONS)

    case_ids = [str(cid) for cid in case_lengths.keys()]
    case_id_set = set(case_ids)

    print("=" * 80)
    print("[Selection] Loading DP edges")
    print("=" * 80)
    print(f"[Selection] edges_path: {edges_path}")
    print(f"[Selection] n_cases in summary: {len(case_ids)}")
    print(f"[Selection] top_k            : {top_k}")
    print(f"[Selection] max_delta        : {max_delta}")
    print(f"[Selection] progress_every   : {progress_every}")

    edges_by_case = load_edges_by_case_original_style(
        edges_path,
        allowed_case_ids=case_id_set,
        progress_every=1_000_000,
    )

    print(f"[Selection] cases with edges: {len(edges_by_case)}")

    rng = random.Random(int(random_seed))

    solution_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []

    solved_cases = 0
    total_input_edges = 0

    for idx, cid in enumerate(case_ids, start=1):
        n = int(case_lengths[cid])

        by_end = edges_by_case.get(cid, {})

        dp_edges_by_i: List[List[Edge]] = [
            []
            for _ in range(n + 1)
        ]

        n_input_edges = 0

        for end_i, elist in by_end.items():
            end_i = int(end_i)

            if 0 <= end_i <= n:
                dp_edges_by_i[end_i] = list(elist)
                n_input_edges += len(elist)

        total_input_edges += n_input_edges

        res = DPResultLite(
            n=n,
            dp_edges_by_i=dp_edges_by_i,
        )

        delta, sols = find_topk_closest_coverage_paths(
            res,
            topk=top_k,
            max_delta=max_delta,
            rng=rng,
        )

        if sols:
            solved_cases += 1

        for rank, sol in enumerate(sols, start=1):
            solution_rows.append(
                solution_to_dict(
                    sol,
                    case_id=cid,
                    case_len=n,
                    rank=rank,
                    delta=delta,
                )
            )

        case_rows.append(
            build_case_output_row(
                case_id=cid,
                case_len=n,
                n_input_edges=n_input_edges,
                delta=delta,
                solutions=sols,
            )
        )

        if idx == 1 or idx % progress_every == 0 or idx == len(case_ids):
            print(
                "[Selection] "
                f"processed {idx}/{len(case_ids)} cases "
                f"| solved={solved_cases} "
                f"| solution_rows={len(solution_rows)} "
                f"| current_case_edges={n_input_edges} "
                f"| total_input_edges={total_input_edges:,}",
                flush=True,
            )

    print("=" * 80)
    print("[Selection] Finished top-k selection")
    print("=" * 80)
    print(f"[Selection] total cases       : {len(case_ids)}")
    print(f"[Selection] solved cases      : {solved_cases}")
    print(f"[Selection] total solution rows: {len(solution_rows)}")
    print(f"[Selection] total input edges : {total_input_edges:,}")

    return solution_rows, case_rows