# src/eae/lpm/extract_paths.py

from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pm4py
from pm4py.objects.process_tree.obj import Operator


def parse_lpm_id(ptml_path: str | Path) -> int:
    """
    Parse LPM id from canonical PTML filename.

    Expected
    --------
    lpm.<id>.ptml
    """
    ptml_path = Path(ptml_path)

    m = re.match(
        r"^lpm\.(\d+)\.ptml$",
        ptml_path.name,
        flags=re.IGNORECASE,
    )

    if not m:
        raise ValueError(f"Unexpected PTML filename: {ptml_path.name}")

    return int(m.group(1))


def load_tree_from_ptml(ptml_path: str | Path):
    """
    Load process tree from PTML.
    """
    return pm4py.read_ptml(str(ptml_path))


def _trim_traces(
    traces: Sequence[Sequence[str]],
    *,
    max_traces: int,
    max_len: int,
) -> List[List[str]]:
    """
    Remove overlong traces and cap the number of traces.
    """
    out = [list(t[:max_len]) for t in traces if len(t) <= max_len]

    if len(out) > max_traces:
        out = out[:max_traces]

    return out


def _dedup_traces(traces: Sequence[Sequence[str]]) -> List[List[str]]:
    """
    Deduplicate traces while preserving order.
    """
    seen = set()
    out = []

    for trace in traces:
        key = tuple(trace)

        if key in seen:
            continue

        seen.add(key)
        out.append(list(trace))

    return out


def _product_concat(
    left: Sequence[Sequence[str]],
    right: Sequence[Sequence[str]],
    cap: int,
) -> List[List[str]]:
    """
    Cartesian product with concatenation.
    """
    out = []

    for a in left:
        for b in right:
            out.append(list(a) + list(b))

            if len(out) >= cap:
                return out

    return out


def _interleave_two(
    a: Sequence[str],
    b: Sequence[str],
    cap: int,
) -> List[List[str]]:
    """
    Enumerate all interleavings of two sequences under a cap.
    """
    out: List[List[str]] = []

    def rec(i: int, j: int, cur: List[str]) -> None:
        if len(out) >= cap:
            return

        if i == len(a) and j == len(b):
            out.append(cur.copy())
            return

        if i < len(a):
            cur.append(str(a[i]))
            rec(i + 1, j, cur)
            cur.pop()

        if j < len(b):
            cur.append(str(b[j]))
            rec(i, j + 1, cur)
            cur.pop()

    rec(0, 0, [])
    return out


def _interleave_many(
    seq_lists: Sequence[Sequence[str]],
    cap: int,
) -> List[List[str]]:
    """
    Enumerate interleavings for multiple sequences progressively.
    """
    if not seq_lists:
        return [[]]

    out = [list(seq_lists[0])]

    for nxt in seq_lists[1:]:
        new_out: List[List[str]] = []

        for base in out:
            inter = _interleave_two(base, nxt, cap - len(new_out))
            new_out.extend(inter)

            if len(new_out) >= cap:
                break

        out = new_out

        if len(out) >= cap:
            break

    return out


def pt_paths(
    tree,
    *,
    loop_max_iters: int = 2,
    max_traces: int = 5000,
    max_len: int = 80,
) -> List[List[str]]:
    """
    Extract bounded executable paths from a PM4Py process tree.

    Supported operators
    -------------------
    - SEQUENCE
    - XOR
    - PARALLEL
    - LOOP

    LOOP rule
    ---------
    PM4Py loops are treated as:
        children[0] = do/body
        children[1] = redo
        children[2] = exit, if present

    The body is executed at least once.
    The redo+body part is repeated up to loop_max_iters - 1 times.
    """

    def collect_nodes(node, acc):
        acc[id(node)] = node
        for child in getattr(node, "children", []) or []:
            collect_nodes(child, acc)

    id2node = {}
    collect_nodes(tree, id2node)

    @lru_cache(maxsize=None)
    def solve(node_id: int, remaining_loop_iters: int) -> Tuple[Tuple[str, ...], ...]:
        node = id2node[node_id]

        if node.operator is None:
            label = getattr(node, "label", None)

            if label is None:
                return (tuple(),)

            return ((str(label),),)

        op = node.operator
        kids = node.children

        if op == Operator.SEQUENCE:
            traces: List[List[str]] = [[]]

            for child in kids:
                child_traces = [list(t) for t in solve(id(child), remaining_loop_iters)]
                traces = _product_concat(traces, child_traces, max_traces)

            return tuple(tuple(t) for t in traces[:max_traces])

        if op == Operator.XOR:
            traces: List[List[str]] = []

            for child in kids:
                traces.extend([list(t) for t in solve(id(child), remaining_loop_iters)])

                if len(traces) >= max_traces:
                    break

            return tuple(tuple(t) for t in traces[:max_traces])

        if op == Operator.PARALLEL:
            child_traces = [
                [list(t) for t in solve(id(child), remaining_loop_iters)]
                for child in kids
            ]

            # Explosion guard: take only a bounded subset per child.
            child_traces = [
                traces[: max(1, max_traces // 10)]
                for traces in child_traces
            ]

            combos: List[List[List[str]]] = [[]]

            for trace_set in child_traces:
                new_combos: List[List[List[str]]] = []

                for base in combos:
                    for t in trace_set:
                        new_combos.append(base + [t])

                        if len(new_combos) >= max_traces:
                            break

                    if len(new_combos) >= max_traces:
                        break

                combos = new_combos

            out: List[List[str]] = []

            for parts in combos:
                inter = _interleave_many(parts, max_traces - len(out))
                out.extend(inter)

                if len(out) >= max_traces:
                    break

            return tuple(tuple(t) for t in out[:max_traces])

        if op == Operator.LOOP:
            do = kids[0]
            redo = kids[1] if len(kids) > 1 else None
            exit_node = kids[2] if len(kids) > 2 else None

            do_traces = [list(t) for t in solve(id(do), remaining_loop_iters)]
            redo_traces = (
                [list(t) for t in solve(id(redo), remaining_loop_iters)]
                if redo is not None
                else [[]]
            )
            exit_traces = (
                [list(t) for t in solve(id(exit_node), remaining_loop_iters)]
                if exit_node is not None
                else [[]]
            )

            out: List[List[str]] = []

            base_once = _product_concat(do_traces, exit_traces, max_traces)
            out.extend(base_once[:max_traces])

            for k in range(1, max(1, loop_max_iters)):
                if len(out) >= max_traces:
                    break

                mid = do_traces

                for _ in range(k):
                    mid = _product_concat(mid, redo_traces, max_traces)
                    mid = _product_concat(mid, do_traces, max_traces)

                cand = _product_concat(mid, exit_traces, max_traces)
                out.extend(cand)
                out = out[:max_traces]

            return tuple(tuple(t) for t in out[:max_traces])

        # Fallback for rare/unsupported operators.
        traces: List[List[str]] = []

        for child in kids:
            traces.extend([list(t) for t in solve(id(child), remaining_loop_iters)])

            if len(traces) >= max_traces:
                break

        return tuple(tuple(t) for t in traces[:max_traces])

    raw = [list(t) for t in solve(id(tree), int(loop_max_iters))]
    raw = _trim_traces(raw, max_traces=max_traces, max_len=max_len)
    raw = _dedup_traces(raw)

    return raw


def build_lpm_groups_payload(
    groups_paths: Sequence[Sequence[str | Path]],
    *,
    ptml_dir: str | Path,
    loop_max_iters: int = 2,
    max_traces_per_lpm: int = 5000,
    max_trace_len: int = 80,
) -> Tuple[Dict[str, object], List[Tuple[str, int | None, str, str]]]:
    """
    Build payload:
        group -> lpm_id -> executable paths

    This corresponds to the original notebook:
        "Build structure: group -> lpm_id -> paths"
    """
    payload: Dict[str, object] = {
        "meta": {
            "pt_dir": str(ptml_dir),
            "created_at": datetime.now().isoformat(),
            "loop_max_iters": int(loop_max_iters),
            "max_traces_per_lpm": int(max_traces_per_lpm),
            "max_trace_len": int(max_trace_len),
        },
        "groups": {},
    }

    build_errors: List[Tuple[str, int | None, str, str]] = []

    groups = payload["groups"]

    for gi, paths in enumerate(groups_paths, start=1):
        gkey = f"group_{gi:02d}"
        groups[gkey] = {}

        for p in paths:
            p = Path(p)

            try:
                lpm_id = parse_lpm_id(p)
                tree = load_tree_from_ptml(p)

                paths_list = pt_paths(
                    tree,
                    loop_max_iters=loop_max_iters,
                    max_traces=max_traces_per_lpm,
                    max_len=max_trace_len,
                )

                groups[gkey][str(lpm_id)] = {
                    "lpm_id": int(lpm_id),
                    "ptml_path": str(p),
                    "n_paths": int(len(paths_list)),
                    "paths": paths_list,
                }

            except Exception as e:
                try:
                    lpm_id_dbg = parse_lpm_id(p)
                except Exception:
                    lpm_id_dbg = None

                build_errors.append((gkey, lpm_id_dbg, str(p), repr(e)))

    return payload, build_errors


def summarize_lpm_groups_payload(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Summarize generated LPM path payload.
    """
    groups = payload.get("groups", {})

    rows = []
    total_lpms = 0
    total_paths = 0

    for gkey, mapping in groups.items():
        n_lpms = len(mapping)
        n_paths_sum = sum(int(v["n_paths"]) for v in mapping.values())
        avg_paths = n_paths_sum / n_lpms if n_lpms else 0.0

        rows.append(
            {
                "group": gkey,
                "n_lpms": n_lpms,
                "total_paths": n_paths_sum,
                "avg_paths_per_lpm": avg_paths,
            }
        )

        total_lpms += n_lpms
        total_paths += n_paths_sum

    return {
        "n_groups": len(groups),
        "n_lpms": total_lpms,
        "n_paths": total_paths,
        "groups": rows,
    }


def save_lpm_groups_payload(
    payload: Dict[str, object],
    output_path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """
    Save LPM group paths payload to JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)

    return output_path


def load_lpm_groups_payload(path: str | Path) -> Dict[str, object]:
    """
    Load LPM group paths payload from JSON.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)