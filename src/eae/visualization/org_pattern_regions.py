from __future__ import annotations

import gzip
import json
import tempfile
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import pm4py
from PIL import Image, ImageDraw

from eae.paths import (
    build_run_dir,
    build_selection_output_paths,
    resolve_path,
)

from .bpmn_comparison import (
    render_bpmn_with_assets,
    resolve_bpmn_model_paths,
)



# ============================================================
# Label-family prefix helpers
# ============================================================

def _normalize_prefixes(
    *values: Any,
) -> tuple[str, ...]:
    prefixes: list[str] = []

    for value in values:
        if value is None:
            continue

        if isinstance(value, (list, tuple, set)):
            candidates = value
        else:
            candidates = [value]

        for candidate in candidates:
            text = str(candidate).strip()

            if not text:
                continue

            if text not in prefixes:
                prefixes.append(text)

    return tuple(prefixes)


def _get_family_prefixes(
    cfg: Dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Red bucket:
        instance/session-based labels.
        Supports current config prefix, plus legacy C/S.

    Blue bucket:
        model/LPM-based labels.
        Supports current config prefix, plus legacy G.
    """
    label_prefix = (
        cfg.get("abstraction_source", {})
        .get("label_prefix", {})
    )

    session_prefix = label_prefix.get("session")
    lpm_prefix = label_prefix.get("lpm")

    red_prefixes = _normalize_prefixes(
        "S",
    )
    blue_prefixes = _normalize_prefixes(
        lpm_prefix,
        "G",
    )

    return red_prefixes, blue_prefixes


def _label_source_bucket(
    label: Any,
    *,
    red_prefixes: tuple[str, ...],
    blue_prefixes: tuple[str, ...],
) -> str | None:
    text = str(label).strip()

    if any(
        text.startswith(prefix)
        for prefix in red_prefixes
        if prefix
    ):
        return "C"

    if any(
        text.startswith(prefix)
        for prefix in blue_prefixes
        if prefix
    ):
        return "G"

    return None


# ============================================================
# Public entry point
# ============================================================

def save_org_pattern_region_overlays(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
    g_max: float = 0.30,
    c_min: float = 0.70,
    min_object_count: int = 1,
) -> Dict[str, Any]:
    """
    Create exactly three persistent ORG-BPMN overlay images:

      - G_region_00_30_object_count.png
      - Neutral_region_30_70_object_count.png
      - C_region_70_100_object_count.png

    C bucket means red/session-side labels.
    G bucket means blue/LPM-side labels.

    Supports both legacy C/G labels and config-based S/G labels.
    """
    if not 0.0 <= g_max <= c_min <= 1.0:
        raise ValueError(
            "Thresholds must satisfy 0 <= g_max <= c_min <= 1."
        )

    red_prefixes, blue_prefixes = _get_family_prefixes(cfg)

    resolved = _resolve_inputs(
        cfg,
        k=k,
        jump=jump,
        rank=rank,
    )

    case_sequences = _load_case_token_sequences(
        resolved["original_log"],
        case_id_column=resolved["case_id_column"],
        activity_column=resolved["activity_column"],
        timestamp_column=resolved["timestamp_column"],
        activity_map=resolved["activity_map"],
    )

    unique_segments_by_label = _extract_unique_used_segments(
        resolved["selected_combinations"],
        case_sequences=case_sequences,
        rank=rank,
        red_prefixes=red_prefixes,
        blue_prefixes=blue_prefixes,
    )

    with tempfile.TemporaryDirectory(
        prefix="eae_org_region_overlay_"
    ) as temp_dir_value:
        temp_dir = Path(temp_dir_value)

        org_assets = render_bpmn_with_assets(
            resolved["org_bpmn"],
            output_png=temp_dir / "org_base.png",
            output_dot=temp_dir / "org_base.dot",
            output_layout_json=temp_dir / "org_layout.json",
            output_meta_json=temp_dir / "org_meta.json",
        )

        layout = _load_json(org_assets["layout_json"])
        meta = _load_json(org_assets["meta_json"])

        matcher = _StructuralBPMNMatcher(
            layout=layout,
            meta=meta,
            activity_map=resolved["activity_map"],
        )

        (
            node_count_c,
            node_count_g,
            edge_count_c,
            edge_count_g,
            match_stats,
        ) = _count_pattern_objects(
            unique_segments_by_label,
            matcher=matcher,
            red_prefixes=red_prefixes,
            blue_prefixes=blue_prefixes,
        )

        c_nodes, n_nodes, g_nodes = _split_object_sets_3way(
            node_count_c,
            node_count_g,
            min_object_count=min_object_count,
            g_max=g_max,
            c_min=c_min,
        )
        c_edges, n_edges, g_edges = _split_object_sets_3way(
            edge_count_c,
            edge_count_g,
            min_object_count=min_object_count,
            g_max=g_max,
            c_min=c_min,
        )

        base_img = Image.open(org_assets["png"]).convert("RGBA")

        output_dir = (
            resolved["run_dir"]
            / "figures"
            / "visualization"
            / "org_pattern_regions"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {
            "g_region_png": (
                output_dir
                / "G_region_00_30_object_count.png"
            ),
            "neutral_region_png": (
                output_dir
                / "Neutral_region_30_70_object_count.png"
            ),
            "c_region_png": (
                output_dir
                / "C_region_70_100_object_count.png"
            ),
        }

        _render_object_region(
            base_img,
            layout=layout,
            meta=meta,
            node_set=g_nodes,
            edge_set=g_edges,
            outline_color=(90, 130, 220, 255),
            fill_color=(185, 205, 245, 145),
            edge_glow_color=(160, 185, 235, 185),
        ).save(output_paths["g_region_png"])

        _render_object_region(
            base_img,
            layout=layout,
            meta=meta,
            node_set=n_nodes,
            edge_set=n_edges,
            outline_color=(70, 150, 90, 255),
            fill_color=(175, 225, 185, 145),
            edge_glow_color=(140, 205, 155, 185),
        ).save(output_paths["neutral_region_png"])

        _render_object_region(
            base_img,
            layout=layout,
            meta=meta,
            node_set=c_nodes,
            edge_set=c_edges,
            outline_color=(190, 70, 70, 255),
            fill_color=(235, 185, 185, 145),
            edge_glow_color=(220, 150, 150, 185),
        ).save(output_paths["c_region_png"])

        base_img.close()

    n_unique_c_segments = int(
        sum(
            len(segments)
            for label, segments in unique_segments_by_label.items()
            if _label_source_bucket(
                label,
                red_prefixes=red_prefixes,
                blue_prefixes=blue_prefixes,
            )
            == "C"
        )
    )

    n_unique_g_segments = int(
        sum(
            len(segments)
            for label, segments in unique_segments_by_label.items()
            if _label_source_bucket(
                label,
                red_prefixes=red_prefixes,
                blue_prefixes=blue_prefixes,
            )
            == "G"
        )
    )

    return {
        **output_paths,
        "output_dir": output_dir,
        "run_dir": resolved["run_dir"],
        "org_bpmn": resolved["org_bpmn"],
        "selected_combinations": resolved["selected_combinations"],
        "activity_map_file": resolved["activity_map_file"],
        "red_prefixes": red_prefixes,
        "blue_prefixes": blue_prefixes,
        "n_unique_c_segments": n_unique_c_segments,
        "n_unique_g_segments": n_unique_g_segments,
        "matched_c_segments": match_stats["matched_c_segments"],
        "matched_g_segments": match_stats["matched_g_segments"],
        "unmatched_c_segments": match_stats["unmatched_c_segments"],
        "unmatched_g_segments": match_stats["unmatched_g_segments"],
        "g_nodes": len(g_nodes),
        "neutral_nodes": len(n_nodes),
        "c_nodes": len(c_nodes),
        "g_edges": len(g_edges),
        "neutral_edges": len(n_edges),
        "c_edges": len(c_edges),
    }


# ============================================================
# Input resolution and loading
# ============================================================

def _resolve_inputs(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int,
) -> Dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    columns_cfg = dataset_cfg["columns"]

    original_log = resolve_path(
        cfg,
        dataset_cfg["raw_log_path"],
    )

    processed_dir = resolve_path(
        cfg,
        dataset_cfg["processed_dir"],
    )
    activity_map_file = (
        processed_dir
        / dataset_cfg.get("activity_map_file", "act2ch.json")
    )

    if not original_log.exists():
        raise FileNotFoundError(
            f"Original XES file does not exist: {original_log}"
        )

    if not activity_map_file.exists():
        raise FileNotFoundError(
            "Activity mapping file does not exist:\n"
            f"{activity_map_file}"
        )

    with activity_map_file.open(
        "r",
        encoding="utf-8",
    ) as handle:
        activity_map = json.load(handle)

    selection_paths = build_selection_output_paths(
        cfg,
        jump=jump,
        k=k,
    )
    selected_combinations = (
        selection_paths["selected_combinations"]
    )

    if not selected_combinations.exists():
        raise FileNotFoundError(
            "Selected-combinations file does not exist. "
            "Run Stage 04 first:\n"
            f"{selected_combinations}"
        )

    bpmn_paths = resolve_bpmn_model_paths(
        cfg,
        k=k,
        jump=jump,
        rank=rank,
    )

    return {
        "run_dir": bpmn_paths["run_dir"],
        "org_bpmn": bpmn_paths["org_bpmn"],
        "original_log": original_log,
        "selected_combinations": selected_combinations,
        "activity_map_file": activity_map_file,
        "activity_map": {
            str(key): str(value)
            for key, value in activity_map.items()
        },
        "case_id_column": columns_cfg["case_id"],
        "activity_column": columns_cfg["activity"],
        "timestamp_column": columns_cfg["timestamp"],
    }


def _load_xes_dataframe(
    xes_path: str | Path,
) -> pd.DataFrame:
    log = pm4py.read_xes(str(xes_path))

    if isinstance(log, pd.DataFrame):
        return log.copy()

    return pm4py.convert_to_dataframe(log)


def _load_case_token_sequences(
    xes_path: str | Path,
    *,
    case_id_column: str,
    activity_column: str,
    timestamp_column: str,
    activity_map: Dict[str, str],
) -> Dict[str, list[str]]:
    frame = _load_xes_dataframe(xes_path)

    required = {
        case_id_column,
        activity_column,
        timestamp_column,
    }
    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            f"Missing XES columns: {sorted(missing)}"
        )

    frame = frame[
        [
            case_id_column,
            activity_column,
            timestamp_column,
        ]
    ].copy()

    frame[case_id_column] = (
        frame[case_id_column].astype(str)
    )
    frame[timestamp_column] = pd.to_datetime(
        frame[timestamp_column],
        errors="coerce",
        utc=True,
    )

    frame = frame.dropna(
        subset=[
            case_id_column,
            activity_column,
            timestamp_column,
        ]
    )

    frame = frame.sort_values(
        [
            case_id_column,
            timestamp_column,
        ],
        kind="mergesort",
    )

    def to_token(activity: Any) -> str:
        raw = str(activity)
        mapped = activity_map.get(raw)
        if mapped is not None:
            return str(mapped)

        # Supports logs that already store one-character tokens.
        if len(raw) == 1:
            return raw

        raise KeyError(
            "Activity is missing from activity map: "
            f"{raw!r}"
        )

    frame["_token"] = frame[activity_column].map(to_token)

    return {
        str(case_id): group["_token"].tolist()
        for case_id, group in frame.groupby(
            case_id_column,
            sort=False,
        )
    }


def _open_jsonl_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _normalize_index(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)

    return None


def _infer_edge_span(
    edge: Dict[str, Any],
    *,
    previous_end: int,
    sequence_length: int,
) -> tuple[int | None, int | None]:
    """
    Support both current [i, j) edges and older reversed i/j exports.
    """
    i_value = _normalize_index(edge.get("i"))
    j_value = _normalize_index(edge.get("j"))

    if (
        i_value is not None
        and j_value is not None
    ):
        if 0 <= i_value < j_value <= sequence_length:
            return i_value, j_value

        if 0 <= j_value < i_value <= sequence_length:
            return j_value, i_value

    start_keys = (
        "start",
        "start_i",
        "start_j",
        "from",
        "left",
    )
    end_keys = (
        "end",
        "end_i",
        "end_j",
        "to",
        "right",
    )

    start = next(
        (
            _normalize_index(edge.get(key))
            for key in start_keys
            if _normalize_index(edge.get(key)) is not None
        ),
        None,
    )
    end = next(
        (
            _normalize_index(edge.get(key))
            for key in end_keys
            if _normalize_index(edge.get(key)) is not None
        ),
        None,
    )

    if end is None:
        return None, None

    if start is None:
        start = previous_end

    start = max(0, min(start, sequence_length))
    end = max(0, min(end, sequence_length))

    if end <= start:
        return None, None

    return start, end



def _extract_unique_used_segments(
    selected_path: str | Path,
    *,
    case_sequences: Dict[str, list[str]],
    rank: int,
    red_prefixes: tuple[str, ...],
    blue_prefixes: tuple[str, ...],
) -> Dict[str, list[list[str]]]:
    """
    Reconstruct unique used segments per selected label.

    Red/session-side labels and blue/LPM-side labels are detected
    from config prefixes with legacy C/S/G support.
    """
    selected_path = Path(selected_path)
    unique_sets: Dict[str, set[tuple[str, ...]]] = defaultdict(set)

    with _open_jsonl_auto(selected_path) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            record_rank = record.get("rank")

            if (
                record_rank is not None
                and int(record_rank) != int(rank)
            ):
                continue

            if record.get("solved") is False:
                continue

            case_id = str(record["case_id"])
            sequence = case_sequences.get(case_id)

            if sequence is None:
                continue

            previous_end = 0

            edges = record.get("edges") or []

            for edge in edges:
                label = edge.get(
                    "label",
                    edge.get("L", edge.get("l")),
                )

                start, end = _infer_edge_span(
                    edge,
                    previous_end=previous_end,
                    sequence_length=len(sequence),
                )

                if end is not None:
                    previous_end = max(
                        previous_end,
                        end,
                    )

                if label is None:
                    continue

                label = str(label).strip()

                if (
                    _label_source_bucket(
                        label,
                        red_prefixes=red_prefixes,
                        blue_prefixes=blue_prefixes,
                    )
                    is None
                ):
                    continue

                if start is None or end is None:
                    continue

                segment = tuple(sequence[start:end])

                if segment:
                    unique_sets[label].add(segment)

    return {
        label: [
            list(segment)
            for segment in sorted(segments)
        ]
        for label, segments in sorted(unique_sets.items())
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


# ============================================================
# Structural BPMN matcher
# ============================================================

class _StructuralBPMNMatcher:
    def __init__(
        self,
        *,
        layout: Dict[str, Any],
        meta: Dict[str, Any],
        activity_map: Dict[str, str],
        max_start_candidates: int = 40,
        max_next_steps: int = 80,
        max_recursion_calls: int = 20000,
        max_found_occurrences: int = 200,
    ) -> None:
        self.layout = layout
        self.meta = meta
        self.activity_map = activity_map

        self.layout_nodes = layout["nodes"]
        self.node_meta = meta["nodes"]
        self.edge_meta = meta["edges"]

        self.max_start_candidates = max_start_candidates
        self.max_next_steps = max_next_steps
        self.max_recursion_calls = max_recursion_calls
        self.max_found_occurrences = max_found_occurrences

        self.out_neighbors: Dict[str, list[str]] = defaultdict(list)

        for edge in self.edge_meta:
            self.out_neighbors[
                edge["source"]
            ].append(edge["target"])

        self.activity_node_ids = [
            node_id
            for node_id, record in self.node_meta.items()
            if record.get("kind") == "activity"
        ]

        self.label_to_activity_nodes: Dict[
            str,
            list[str],
        ] = defaultdict(list)

        for node_id in self.activity_node_ids:
            token = self._node_label_token(node_id)
            if token is not None:
                self.label_to_activity_nodes[
                    token
                ].append(node_id)

    @staticmethod
    def _clean_raw_label(raw: Any) -> str:
        text = str(raw).strip()

        for stop in (
            " fontsize=",
            " shape=",
            " style=",
            " fillcolor=",
            " color=",
            " penwidth=",
            " fontname=",
        ):
            if stop in text:
                text = text.split(stop, 1)[0].strip()

        if (
            len(text) >= 2
            and text[0] == '"'
            and text[-1] == '"'
        ):
            text = text[1:-1].strip()

        return text

    def _node_label_token(
        self,
        node_id: str,
    ) -> str | None:
        raw = self._clean_raw_label(
            self.node_meta[node_id].get("label", "")
        )

        mapped = self.activity_map.get(raw)

        if mapped is not None:
            return str(mapped)

        if len(raw) == 1:
            return raw

        return None

    def _is_visible_activity(
        self,
        node_id: str,
    ) -> bool:
        return (
            self.node_meta[node_id].get("kind")
            == "activity"
            and self._node_label_token(node_id)
            is not None
        )

    @lru_cache(maxsize=None)
    def _activity_depth_score(
        self,
        node_id: str,
    ) -> tuple[float, float]:
        record = self.layout_nodes.get(node_id, {})
        x = float(record.get("x", 0.0))
        y = float(record.get("y", 0.0))
        return (-x, -y)

    def _rank_start_candidates(
        self,
        node_ids: Iterable[str],
    ) -> list[str]:
        return sorted(
            node_ids,
            key=self._activity_depth_score,
        )[: self.max_start_candidates]

    @lru_cache(maxsize=None)
    def _next_visible_steps_from(
        self,
        start_node: str,
    ) -> tuple[Dict[str, Any], ...]:
        output: list[Dict[str, Any]] = []
        queue = deque()

        for next_node in self.out_neighbors.get(
            start_node,
            [],
        ):
            queue.append(
                (
                    next_node,
                    frozenset({next_node}),
                    frozenset(
                        {(start_node, next_node)}
                    ),
                )
            )

        visited = set()

        while (
            queue
            and len(output) < self.max_next_steps
        ):
            current, node_set, edge_set = (
                queue.popleft()
            )

            state_key = (current, node_set)

            if state_key in visited:
                continue

            visited.add(state_key)

            if self._is_visible_activity(current):
                output.append(
                    {
                        "next_activity": current,
                        "node_set": node_set,
                        "edge_set": edge_set,
                    }
                )
                continue

            for next_node in self.out_neighbors.get(
                current,
                [],
            ):
                queue.append(
                    (
                        next_node,
                        frozenset(
                            set(node_set)
                            | {next_node}
                        ),
                        frozenset(
                            set(edge_set)
                            | {(current, next_node)}
                        ),
                    )
                )

        return tuple(output)

    def find_segment(
        self,
        segment_tokens: list[str],
    ) -> Dict[str, Any] | None:
        if not segment_tokens:
            return None

        first_token = str(
            segment_tokens[0]
        ).strip()

        start_candidates = (
            self.label_to_activity_nodes.get(
                first_token,
                [],
            )
        )

        if not start_candidates:
            return None

        start_candidates = (
            self._rank_start_candidates(
                start_candidates
            )
        )

        results: list[Dict[str, Any]] = []
        recursion_calls = 0
        hard_stop = False

        def recurse(
            current_activity: str,
            token_index: int,
            matched_activity_ids: list[str],
            node_set: set[str],
            edge_set: set[tuple[str, str]],
        ) -> None:
            nonlocal recursion_calls, hard_stop

            if hard_stop:
                return

            recursion_calls += 1

            if (
                recursion_calls
                > self.max_recursion_calls
            ):
                hard_stop = True
                return

            if (
                len(results)
                >= self.max_found_occurrences
            ):
                hard_stop = True
                return

            if token_index == len(segment_tokens):
                results.append(
                    {
                        "matched_activity_ids": tuple(
                            matched_activity_ids
                        ),
                        "node_set": set(node_set),
                        "edge_set": set(edge_set),
                    }
                )
                return

            target_token = str(
                segment_tokens[token_index]
            ).strip()

            for step in self._next_visible_steps_from(
                current_activity
            ):
                next_activity = step[
                    "next_activity"
                ]

                if (
                    self._node_label_token(
                        next_activity
                    )
                    != target_token
                ):
                    continue

                recurse(
                    next_activity,
                    token_index + 1,
                    matched_activity_ids
                    + [next_activity],
                    node_set
                    | set(step["node_set"]),
                    edge_set
                    | set(step["edge_set"]),
                )

                if hard_stop:
                    return

        for start_node in start_candidates:
            recurse(
                start_node,
                1,
                [start_node],
                {start_node},
                set(),
            )

            if hard_stop and results:
                break

        if not results:
            return None

        # Prefer the structurally smallest occurrence.
        return min(
            results,
            key=lambda result: (
                len(result["node_set"]),
                len(result["edge_set"]),
                -len(
                    result[
                        "matched_activity_ids"
                    ]
                ),
            ),
        )


# ============================================================
# Counting and three-way split
# ============================================================


def _count_pattern_objects(
    unique_segments_by_label: Dict[
        str,
        list[list[str]],
    ],
    *,
    matcher: _StructuralBPMNMatcher,
    red_prefixes: tuple[str, ...],
    blue_prefixes: tuple[str, ...],
) -> tuple[
    Counter,
    Counter,
    Counter,
    Counter,
    Dict[str, int],
]:
    node_count_c: Counter = Counter()
    node_count_g: Counter = Counter()
    edge_count_c: Counter = Counter()
    edge_count_g: Counter = Counter()

    stats = {
        "matched_c_segments": 0,
        "matched_g_segments": 0,
        "unmatched_c_segments": 0,
        "unmatched_g_segments": 0,
    }

    for label, segments in unique_segments_by_label.items():
        source = _label_source_bucket(
            label,
            red_prefixes=red_prefixes,
            blue_prefixes=blue_prefixes,
        )

        if source is None:
            continue

        for segment in segments:
            result = matcher.find_segment(segment)

            if result is None:
                stats[
                    f"unmatched_{source.lower()}_segments"
                ] += 1
                continue

            stats[
                f"matched_{source.lower()}_segments"
            ] += 1

            if source == "C":
                node_count_c.update(
                    result["node_set"]
                )
                edge_count_c.update(
                    result["edge_set"]
                )
            else:
                node_count_g.update(
                    result["node_set"]
                )
                edge_count_g.update(
                    result["edge_set"]
                )

    return (
        node_count_c,
        node_count_g,
        edge_count_c,
        edge_count_g,
        stats,
    )


def _split_object_sets_3way(
    count_c: Counter,
    count_g: Counter,
    *,
    min_object_count: int,
    g_max: float,
    c_min: float,
) -> tuple[set, set, set]:
    c_set = set()
    neutral_set = set()
    g_set = set()

    keys = set(count_c) | set(count_g)

    for key in keys:
        c_count = int(count_c.get(key, 0))
        g_count = int(count_g.get(key, 0))
        total = c_count + g_count

        if total < min_object_count:
            continue

        ratio_c = c_count / total

        if ratio_c < g_max:
            g_set.add(key)
        elif ratio_c < c_min:
            neutral_set.add(key)
        else:
            c_set.add(key)

    return c_set, neutral_set, g_set


# ============================================================
# Rendering
# ============================================================

def _cubic_bezier(
    p0,
    p1,
    p2,
    p3,
    t,
):
    x = (
        (1 - t) ** 3 * p0[0]
        + 3 * (1 - t) ** 2 * t * p1[0]
        + 3 * (1 - t) * t**2 * p2[0]
        + t**3 * p3[0]
    )
    y = (
        (1 - t) ** 3 * p0[1]
        + 3 * (1 - t) ** 2 * t * p1[1]
        + 3 * (1 - t) * t**2 * p2[1]
        + t**3 * p3[1]
    )
    return x, y


def _sample_cubic_chain(
    control_points,
    *,
    samples_per_segment: int = 24,
):
    if len(control_points) < 4:
        return control_points[:]

    output = []
    start = 0
    first = True

    while start + 3 < len(control_points):
        p0, p1, p2, p3 = (
            control_points[start : start + 4]
        )

        points = [
            _cubic_bezier(
                p0,
                p1,
                p2,
                p3,
                i / samples_per_segment,
            )
            for i in range(
                samples_per_segment + 1
            )
        ]

        output.extend(
            points
            if first
            else points[1:]
        )

        first = False
        start += 3

    if start < len(control_points) - 1:
        output.extend(
            control_points[start + 1 :]
        )

    return output


def _render_object_region(
    base_img: Image.Image,
    *,
    layout: Dict[str, Any],
    meta: Dict[str, Any],
    node_set: set,
    edge_set: set,
    outline_color,
    fill_color,
    edge_glow_color,
    edge_width: int = 5,
    node_width: int = 4,
    supersample: int = 2,
) -> Image.Image:
    width, height = base_img.size

    layout_nodes = layout["nodes"]
    layout_edge_map = {
        (edge["tail"], edge["head"]): edge
        for edge in layout.get("edges", [])
    }
    node_meta = meta["nodes"]

    graph_width = float(
        layout["graph"]["width"]
    )
    graph_height = float(
        layout["graph"]["height"]
    )

    big_base = base_img.resize(
        (
            width * supersample,
            height * supersample,
        ),
        Image.Resampling.LANCZOS,
    )

    overlay = Image.new(
        "RGBA",
        big_base.size,
        (255, 255, 255, 0),
    )
    draw = ImageDraw.Draw(overlay, "RGBA")

    def graph_to_pixel(x, y):
        return (
            (float(x) / graph_width)
            * (width * supersample),
            (height * supersample)
            - (float(y) / graph_height)
            * (height * supersample),
        )

    def graph_width_to_pixel(value):
        return (
            float(value)
            / graph_width
            * (width * supersample)
        )

    def graph_height_to_pixel(value):
        return (
            float(value)
            / graph_height
            * (height * supersample)
        )

    for edge_key in edge_set:
        edge_layout = layout_edge_map.get(
            edge_key
        )

        if edge_layout is None:
            continue

        points = edge_layout.get("points", [])

        if len(points) < 2:
            continue

        control_points = [
            graph_to_pixel(x, y)
            for x, y in points
        ]

        curve = _sample_cubic_chain(
            control_points
        )

        draw.line(
            curve,
            fill=edge_glow_color,
            width=max(
                1,
                (edge_width + 3)
                * supersample,
            ),
        )
        draw.line(
            curve,
            fill=outline_color,
            width=max(
                1,
                edge_width * supersample,
            ),
        )

    for node_id in node_set:
        if (
            node_id not in layout_nodes
            or node_id not in node_meta
        ):
            continue

        node_layout = layout_nodes[node_id]
        node_record = node_meta[node_id]

        center_x, center_y = graph_to_pixel(
            node_layout["x"],
            node_layout["y"],
        )

        node_w = graph_width_to_pixel(
            node_layout["width"]
        )
        node_h = graph_height_to_pixel(
            node_layout["height"]
        )

        box = [
            center_x - node_w / 2,
            center_y - node_h / 2,
            center_x + node_w / 2,
            center_y + node_h / 2,
        ]

        line_width = max(
            1,
            node_width * supersample,
        )

        kind = node_record.get(
            "kind",
            "other",
        )

        if kind == "activity":
            draw.rounded_rectangle(
                box,
                radius=max(
                    4,
                    int(10 * supersample),
                ),
                fill=fill_color,
                outline=outline_color,
                width=line_width,
            )
        elif kind == "gateway":
            draw.polygon(
                [
                    (
                        center_x,
                        center_y - node_h / 2,
                    ),
                    (
                        center_x + node_w / 2,
                        center_y,
                    ),
                    (
                        center_x,
                        center_y + node_h / 2,
                    ),
                    (
                        center_x - node_w / 2,
                        center_y,
                    ),
                ],
                fill=fill_color,
                outline=outline_color,
                width=line_width,
            )
        else:
            draw.ellipse(
                box,
                fill=fill_color,
                outline=outline_color,
                width=line_width,
            )

    merged = Image.alpha_composite(
        big_base,
        overlay,
    )

    result = merged.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    overlay.close()
    big_base.close()
    merged.close()

    return result
