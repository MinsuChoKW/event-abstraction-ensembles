# src/eae/discovery/log_builder.py

from __future__ import annotations

import ast
import gzip
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pm4py


def seqstr_to_events(value: Any) -> List[str]:
    """
    Convert a stored sequence value to an event-token list.

    Supports:
      - list / tuple
      - JSON list string
      - Python tuple/list string
      - whitespace-separated string
      - single token string

    Unlike the old BPI2012 alpha-string loader, a single string is not
    split character by character.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [
            str(x)
            for x in value
            if str(x) != ""
        ]

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        if (
            (text.startswith("[") and text.endswith("]"))
            or
            (text.startswith("(") and text.endswith(")"))
        ):
            try:
                parsed = json.loads(text)

                if isinstance(parsed, (list, tuple)):
                    return [
                        str(x)
                        for x in parsed
                        if str(x) != ""
                    ]
            except Exception:
                try:
                    parsed = ast.literal_eval(text)

                    if isinstance(parsed, (list, tuple)):
                        return [
                            str(x)
                            for x in parsed
                            if str(x) != ""
                        ]
                except Exception:
                    pass

        if " " in text:
            return [
                token
                for token in text.split()
                if token
            ]

        return [text]

    return [str(value)]


def canonicalize_traces(
    traces: Iterable[Any],
) -> List[List[str]]:
    """
    Remove empty and duplicate traces, then sort deterministically.

    Original notebook ordering:
      1. trace length
      2. lexicographic token sequence
    """
    unique: Dict[Tuple[str, ...], List[str]] = {}

    for trace in traces or []:
        events = seqstr_to_events(trace)

        if not events:
            continue

        key = tuple(
            str(x)
            for x in events
            if str(x) != ""
        )

        if key:
            unique[key] = list(key)

    return [
        unique[key]
        for key in sorted(
            unique.keys(),
            key=lambda x: (len(x), x),
        )
    ]


def filter_empty(
    case_ids: List[str],
    traces: List[List[str]],
) -> Tuple[List[str], List[List[str]]]:
    new_case_ids: List[str] = []
    new_traces: List[List[str]] = []

    for case_id, trace in zip(case_ids, traces):
        if trace:
            new_case_ids.append(str(case_id))
            new_traces.append(list(trace))

    return new_case_ids, new_traces


def load_case_strings_jsonl_gz(
    path: str | Path,
    *,
    limit: Optional[int] = None,
) -> "OrderedDict[str, List[str]]":
    """
    Load processed case strings.

    Supported sequence fields:
      sequence
      s
      events
      activities
      trace
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Case strings file does not exist: {path}"
        )

    out: "OrderedDict[str, List[str]]" = OrderedDict()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break

            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            case_id = str(
                obj.get(
                    "case_id",
                    obj.get("case:concept:name", idx),
                )
            )

            sequence_value = (
                obj.get("sequence")
                or obj.get("s")
                or obj.get("events")
                or obj.get("activities")
                or obj.get("trace")
                or []
            )

            out[case_id] = seqstr_to_events(sequence_value)

    return out


def read_jsonl_gz(
    path: str | Path,
) -> Iterable[Dict[str, Any]]:
    path = Path(path)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                yield json.loads(line)


def load_selected_rank_solutions(
    path: str | Path,
    *,
    rank: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Load rank-specific solution rows produced by stage 04.

    Current stage-04 format is one solution per JSONL line:
      {
        "case_id": ...,
        "rank": 1,
        "end_i": ...,
        "edges": [...]
      }
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Selected-combination file does not exist: {path}"
        )

    out: Dict[str, Dict[str, Any]] = {}

    for row in read_jsonl_gz(path):
        row_rank = int(row.get("rank", -1))

        if row_rank != int(rank):
            continue

        case_id = str(row["case_id"])

        normalized = dict(row)
        normalized["solved"] = bool(
            normalized.get("edges", [])
        )

        out[case_id] = normalized

    return out


def extract_label_from_edge(
    edge: Dict[str, Any],
) -> Optional[str]:
    label = edge.get(
        "label",
        edge.get("L"),
    )

    if label is None:
        return None

    label = str(label).strip()

    return label if label else None


def build_logs_from_selected_rank(
    *,
    case_sequence_map: Dict[str, List[str]],
    selected_solution_map: Dict[str, Dict[str, Any]],
    use_prefix_end_i: bool = True,
    require_all_cases: bool = False,
) -> Tuple[
    List[str],
    List[List[str]],
    List[List[str]],
    List[str],
]:
    """
    Original notebook build_logs_from_top1 logic.

    Returns
    -------
    case_ids
        Cases included in both logs.

    traces_original
        Original low-level traces, optionally truncated at end_i.

    traces_abstract
        Abstract label sequences generated from selected edges.

    missing_case_ids
        Cases without a usable selected solution.
    """
    case_ids: List[str] = []
    traces_original: List[List[str]] = []
    traces_abstract: List[List[str]] = []
    missing: List[str] = []

    for case_id, sequence in case_sequence_map.items():
        solution = selected_solution_map.get(str(case_id))

        if solution is None:
            missing.append(str(case_id))
            continue

        edges = solution.get("edges", []) or []

        if not edges:
            missing.append(str(case_id))
            continue

        abstract_tokens: List[str] = []
        last_end = 0

        for edge in edges:
            label = extract_label_from_edge(edge)

            if label:
                abstract_tokens.append(label)

            # Current project convention:
            #   edge["i"] = start
            #   edge["j"] = end
            end_position = edge.get(
                "j",
                edge.get("end_i"),
            )

            if isinstance(end_position, int):
                last_end = max(last_end, end_position)

        if not abstract_tokens:
            missing.append(str(case_id))
            continue

        original_sequence = list(sequence)

        solution_end = solution.get(
            "end_i",
            solution.get("terminal_pos"),
        )

        if (
            use_prefix_end_i
            and isinstance(solution_end, int)
        ):
            end_i = max(
                0,
                min(
                    int(solution_end),
                    len(original_sequence),
                ),
            )
        else:
            end_i = (
                last_end
                if last_end > 0
                else len(original_sequence)
            )

        original_prefix = original_sequence[:end_i]

        if not original_prefix:
            missing.append(str(case_id))
            continue

        case_ids.append(str(case_id))
        traces_original.append(original_prefix)
        traces_abstract.append(abstract_tokens)

    if require_all_cases and missing:
        raise ValueError(
            f"Missing or unsolved cases: {len(missing)}. "
            f"Examples: {missing[:10]}"
        )

    case_ids_original, traces_original = filter_empty(
        case_ids,
        traces_original,
    )

    case_ids_abstract, traces_abstract = filter_empty(
        case_ids,
        traces_abstract,
    )

    if case_ids_original != case_ids_abstract:
        raise ValueError(
            "Original and abstracted case sets differ."
        )

    return (
        case_ids_original,
        traces_original,
        traces_abstract,
        missing,
    )


def build_pm4py_log_from_traces(
    traces: List[List[str]],
    case_ids: List[str],
    *,
    stable_sort: bool = True,
):
    """
    Original notebook PM4Py log builder.

    Stable sorting is used so that repeated runs receive traces in a
    deterministic order.
    """
    from pm4py.objects.conversion.log import (
        converter as log_converter,
    )

    pairs = [
        (
            str(case_id),
            [
                str(event)
                for event in trace
                if str(event) != ""
            ],
        )
        for case_id, trace in zip(case_ids, traces)
    ]

    pairs = [
        pair
        for pair in pairs
        if pair[1]
    ]

    if stable_sort:
        pairs = sorted(
            pairs,
            key=lambda x: (
                tuple(x[1]),
                x[0],
            ),
        )

    rows: List[Dict[str, Any]] = []

    base_time = pd.Timestamp("2020-01-01")
    global_index = 0

    for case_id, events in pairs:
        for event in events:
            rows.append(
                {
                    "case:concept:name": case_id,
                    "concept:name": str(event),
                    "time:timestamp": (
                        base_time
                        + pd.Timedelta(
                            seconds=global_index
                        )
                    ),
                }
            )

            global_index += 1

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "Event log dataframe is empty."
        )

    df = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    return log_converter.apply(
        df,
        variant=log_converter.Variants.TO_EVENT_LOG,
    )


def export_xes_log(
    log,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pm4py.write_xes(
        log,
        str(path),
    )

    return path