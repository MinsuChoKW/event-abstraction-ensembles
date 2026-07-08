# src/eae/discovery/local_models.py

from __future__ import annotations

import gzip
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from eae.discovery.log_builder import (
    build_pm4py_log_from_traces,
    canonicalize_traces,
)
from eae.discovery.model_discovery_algo import (
    discover_stable_model,
    save_petri_net,
    save_process_tree,
)


def load_pattern_pool_by_label(
    path: str | Path,
) -> "OrderedDict[str, List[List[str]]]":
    """
    Load stage-03 pattern pool.

    Input:
      pattern_pool/pattern_pool_by_label.json.gz
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Pattern pool does not exist: {path}"
        )

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    out: "OrderedDict[str, List[List[str]]]" = (
        OrderedDict()
    )

    for label in sorted(
        payload.keys(),
        key=str,
    ):
        out[str(label)] = canonicalize_traces(
            payload.get(label, [])
        )

    return out


def safe_label_filename(
    label: str,
) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(label),
    )


def discover_local_models_from_pattern_pool(
    pattern_pool_by_label: Dict[
        str,
        List[List[str]],
    ],
    *,
    local_models_dir: str | Path,
    noise_threshold: float = 0.0,
    progress_every: int = 10,
) -> Tuple[
    Dict[str, Tuple[Any, Any, Any]],
    pd.DataFrame,
]:
    """
    Original notebook local-model construction.

    For each label:
      1. use the label's path set as local traces;
      2. build a PM4Py local event log;
      3. discover a process tree with IM;
      4. convert it to a local Petri net;
      5. save PTML, tree text, PNML and metadata.
    """
    local_models_dir = Path(
        local_models_dir
    )
    local_models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = sorted(
        pattern_pool_by_label.keys()
    )

    local_models: Dict[
        str,
        Tuple[Any, Any, Any],
    ] = {}

    rows: List[Dict[str, Any]] = []

    n_labels = len(labels)

    for idx, label in enumerate(
        labels,
        start=1,
    ):
        paths = canonicalize_traces(
            pattern_pool_by_label[label]
        )

        safe_label = safe_label_filename(
            label
        )

        label_dir = (
            local_models_dir / safe_label
        )
        label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not paths:
            rows.append(
                {
                    "label": label,
                    "status": "empty",
                    "n_paths": 0,
                    "error": None,
                }
            )
            continue

        case_ids = [
            f"{label}__c{i:04d}"
            for i in range(
                1,
                len(paths) + 1,
            )
        ]

        try:
            local_log = build_pm4py_log_from_traces(
                paths,
                case_ids,
                stable_sort=True,
            )

            (
                tree,
                net,
                initial_marking,
                final_marking,
            ) = discover_stable_model(
                local_log,
                noise_threshold=(
                    noise_threshold
                ),
            )

            tree_paths = save_process_tree(
                tree,
                text_path=(
                    label_dir
                    / f"{safe_label}.tree.txt"
                ),
                ptml_path=(
                    label_dir
                    / f"{safe_label}.ptml"
                ),
            )

            net_meta = save_petri_net(
                net,
                initial_marking,
                final_marking,
                pnml_path=(
                    label_dir
                    / f"{safe_label}.pnml"
                ),
                meta_path=(
                    label_dir
                    / f"{safe_label}.net_meta.json"
                ),
            )

            local_models[label] = (
                net,
                initial_marking,
                final_marking,
            )

            rows.append(
                {
                    "label": label,
                    "status": "ok",
                    "n_paths": int(len(paths)),
                    "min_path_len": int(
                        min(map(len, paths))
                    ),
                    "max_path_len": int(
                        max(map(len, paths))
                    ),
                    "tree_text": (
                        tree_paths[
                            "tree_text_path"
                        ]
                    ),
                    "tree_ptml": (
                        tree_paths[
                            "ptml_path"
                        ]
                    ),
                    "petri_pnml": (
                        net_meta["pnml_path"]
                    ),
                    "places": (
                        net_meta["places"]
                    ),
                    "transitions": (
                        net_meta[
                            "transitions"
                        ]
                    ),
                    "arcs": net_meta["arcs"],
                    "error": None,
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "label": label,
                    "status": "error",
                    "n_paths": int(len(paths)),
                    "error": repr(exc),
                }
            )

        if (
            idx == 1
            or idx % progress_every == 0
            or idx == n_labels
        ):
            n_ok = sum(
                row["status"] == "ok"
                for row in rows
            )

            print(
                "[LocalModels] "
                f"processed {idx}/{n_labels} labels "
                f"| successful={n_ok}",
                flush=True,
            )

    return (
        local_models,
        pd.DataFrame(rows),
    )