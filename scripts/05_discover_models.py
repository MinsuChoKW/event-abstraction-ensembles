# scripts/05_discover_models.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from eae.config import (
    load_config,
)
from eae.paths import (
    build_case_strings_path,
    build_run_dir,
    create_run_subdirs,
)

from eae.discovery.log_builder import (
    build_logs_from_selected_rank,
    build_pm4py_log_from_traces,
    export_xes_log,
    load_case_strings_jsonl_gz,
    load_selected_rank_solutions,
)

from eae.discovery.model_discovery_algo import (
    discover_stable_model,
    load_petri_net,
    save_petri_net,
    save_process_tree,
)

from eae.discovery.local_models import (
    discover_local_models_from_pattern_pool,
    load_pattern_pool_by_label,
)

from eae.discovery.expansion import (
    expand_abstract_petri_net,
    save_expanded_model,
)


def save_json(
    path: str | Path,
    obj: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path

def load_json(
    path: str | Path,
) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def float_equal(
    value_a: Any,
    value_b: Any,
    *,
    tolerance: float = 1e-12,
) -> bool:
    try:
        return abs(
            float(value_a) - float(value_b)
        ) <= tolerance
    except (TypeError, ValueError):
        return False


def can_reuse_abs_model(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
    *,
    k: int,
    jump: int,
    rank: int,
    abstract_noise: float,
) -> tuple[bool, str]:
    """
    Existing ABS model is always reused when all discovery
    conditions match the current target configuration.
    """
    required_files = [
        paths["abstract_pnml"],
        paths["abstract_tree_ptml"],
        paths["abstract_meta"],
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        return (
            False,
            "missing required files: "
            + ", ".join(str(path) for path in missing_files),
        )

    try:
        metadata = load_json(
            paths["abstract_meta"]
        )
    except Exception as exc:
        return (
            False,
            f"failed to read ABS metadata: {exc}",
        )

    expected = {
        "model_type": "ABS",
        "dataset": str(cfg["dataset"]["name"]),
        "method": str(
            cfg["abstraction_source"]["method"]
        ),
        "K": int(k),
        "jump": int(jump),
        "rank": int(rank),
    }

    for key, expected_value in expected.items():
        saved_value = metadata.get(key)

        if str(saved_value) != str(expected_value):
            return (
                False,
                f"{key} mismatch: "
                f"saved={saved_value}, "
                f"requested={expected_value}",
            )

    saved_noise = metadata.get(
        "noise_threshold"
    )

    if not float_equal(
        saved_noise,
        abstract_noise,
    ):
        return (
            False,
            "abstract-noise mismatch: "
            f"saved={saved_noise}, "
            f"requested={abstract_noise}",
        )

    return (
        True,
        "all ABS discovery conditions match",
    )

def build_discovery_paths(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int,
) -> Dict[str, Path]:
    run_dir = build_run_dir(
        cfg,
        jump=jump,
        k=k,
    )

    create_run_subdirs(
        cfg,
        run_dir,
    )

    pattern_pool_dir = (
        run_dir / "pattern_pool"
    )
    selection_dir = (
        run_dir / "selection"
    )
    discovery_dir = (
        run_dir / "discovery"
    )

    discovery_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_models_dir = (
        discovery_dir / "local_models"
    )

    local_models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "run_dir": run_dir,
        "pattern_pool": (
            pattern_pool_dir
            / "pattern_pool_by_label.json.gz"
        ),
        "selected_combinations": (
            selection_dir
            / "selected_combinations_top5.jsonl.gz"
        ),
        "case_strings": (
            build_case_strings_path(cfg)
        ),

        "abstracted_xes": (
            discovery_dir
            / f"abstracted_log_rank{rank}.xes"
        ),
        "original_prefix_xes": (
            discovery_dir
            / f"original_prefix_log_rank{rank}.xes"
        ),

        "abstract_tree_text": (
            discovery_dir
            / f"abstract_model_rank{rank}.tree.txt"
        ),
        "abstract_tree_ptml": (
            discovery_dir
            / f"abstract_model_rank{rank}.ptml"
        ),
        "abstract_pnml": (
            discovery_dir
            / f"abstract_model_rank{rank}.pnml"
        ),
        "abstract_meta": (
            discovery_dir
            / f"abstract_model_rank{rank}.net_meta.json"
        ),

        "local_models_dir": (
            local_models_dir
        ),
        "local_models_summary": (
            discovery_dir
            / f"local_models_summary_rank{rank}.csv"
        ),

        "expanded_pnml": (
            discovery_dir
            / f"expanded_model_rank{rank}.pnml"
        ),
        "expanded_meta": (
            discovery_dir
            / f"expanded_model_rank{rank}.json"
        ),

        "stage_report": (
            discovery_dir
            / f"stage05_report_rank{rank}.json"
        ),
    }


def run_discovery_for_setting(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
    abstract_noise: float = 0.20,
    local_noise: float = 0.0,
    progress_every: int = 10,
) -> Dict[str, Any]:
    paths = build_discovery_paths(
        cfg,
        k=k,
        jump=jump,
        rank=rank,
    )

    print("=" * 80)
    print("[Stage05] Start ABS and EXP model construction")
    print("=" * 80)
    print(f"[Stage05] dataset       : {cfg['dataset']['name']}")
    print(f"[Stage05] method        : {cfg['abstraction_source']['method']}")
    print(f"[Stage05] K             : {k}")
    print(f"[Stage05] jump          : {jump}")
    print(f"[Stage05] rank          : {rank}")
    print(f"[Stage05] abstract noise: {abstract_noise}")
    print(f"[Stage05] local noise   : {local_noise}")
    print(f"[Stage05] run_dir       : {paths['run_dir']}")

    required_inputs = [
        paths["case_strings"],
        paths["selected_combinations"],
        paths["pattern_pool"],
    ]

    for input_path in required_inputs:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Required stage-05 input is missing: {input_path}"
            )

    # --------------------------------------------------------
    # 1. Load cases and selected rank
    # --------------------------------------------------------
    print("[Stage05:1] Loading cases and selected solutions")

    case_sequence_map = (
        load_case_strings_jsonl_gz(
            paths["case_strings"]
        )
    )

    selected_solution_map = (
        load_selected_rank_solutions(
            paths["selected_combinations"],
            rank=rank,
        )
    )

    (
        case_ids,
        original_prefix_traces,
        abstract_traces,
        missing_cases,
    ) = build_logs_from_selected_rank(
        case_sequence_map=(
            case_sequence_map
        ),
        selected_solution_map=(
            selected_solution_map
        ),
        use_prefix_end_i=True,
        require_all_cases=False,
    )

    if not case_ids:
        raise ValueError(
            "No usable selected solutions were found."
        )

    print(
        f"[Stage05:1] cases={len(case_ids)} "
        f"| original events="
        f"{sum(map(len, original_prefix_traces))} "
        f"| abstract events="
        f"{sum(map(len, abstract_traces))} "
        f"| missing={len(missing_cases)}"
    )

    original_log = build_pm4py_log_from_traces(
        original_prefix_traces,
        case_ids,
        stable_sort=True,
    )

    abstract_log = build_pm4py_log_from_traces(
        abstract_traces,
        case_ids,
        stable_sort=True,
    )

    export_xes_log(
        original_log,
        paths["original_prefix_xes"],
    )

    export_xes_log(
        abstract_log,
        paths["abstracted_xes"],
    )

    # --------------------------------------------------------
    # 2. Load or discover ABS model
    # --------------------------------------------------------
    print("[Stage05:2] Preparing abstract high-level model")

    can_reuse_abs, reuse_reason = can_reuse_abs_model(
        cfg,
        paths,
        k=k,
        jump=jump,
        rank=rank,
        abstract_noise=abstract_noise,
    )

    if can_reuse_abs:
        # ----------------------------------------------------
        # Reuse existing ABS model
        # ----------------------------------------------------
        print(
            "[Stage05:2] Matching ABS model already exists."
        )
        print(
            "[Stage05:2] Skipping ABS discovery."
        )
        print(
            "[Stage05:2] reason:",
            reuse_reason,
        )
        print(
            "[Stage05:2] PNML:",
            paths["abstract_pnml"],
        )

        (
            abstract_net,
            abstract_initial_marking,
            abstract_final_marking,
        ) = load_petri_net(
            paths["abstract_pnml"]
        )

        abstract_net_meta = load_json(
            paths["abstract_meta"]
        )

        abstract_tree_outputs = {
            "tree_text_path": str(
                paths["abstract_tree_text"]
            ),
            "ptml_path": str(
                paths["abstract_tree_ptml"]
            ),
            "ptml_error": None,
        }

        abs_created_this_run = False

    else:
        # ----------------------------------------------------
        # Discover new ABS model
        # ----------------------------------------------------
        print(
            "[Stage05:2] No reusable ABS model."
        )
        print(
            "[Stage05:2] reason:",
            reuse_reason,
        )
        print(
            "[Stage05:2] Discovering new ABS model."
        )

        (
            abstract_tree,
            abstract_net,
            abstract_initial_marking,
            abstract_final_marking,
        ) = discover_stable_model(
            abstract_log,
            noise_threshold=abstract_noise,
        )

        abstract_tree_outputs = save_process_tree(
            abstract_tree,
            text_path=(
                paths["abstract_tree_text"]
            ),
            ptml_path=(
                paths["abstract_tree_ptml"]
            ),
        )

        base_net_meta = save_petri_net(
            abstract_net,
            abstract_initial_marking,
            abstract_final_marking,
            pnml_path=(
                paths["abstract_pnml"]
            ),
        )

        abstract_net_meta = {
            **base_net_meta,
            "model_type": "ABS",
            "dataset": str(
                cfg["dataset"]["name"]
            ),
            "method": str(
                cfg[
                    "abstraction_source"
                ]["method"]
            ),
            "K": int(k),
            "jump": int(jump),
            "rank": int(rank),
            "noise_threshold": float(
                abstract_noise
            ),
        }

        save_json(
            paths["abstract_meta"],
            abstract_net_meta,
        )

        abs_created_this_run = True

    print(
        "[Stage05:2] ABS model:",
        abstract_net_meta,
    )

    # --------------------------------------------------------
    # 3. Discover local models
    # --------------------------------------------------------
    print("[Stage05:3] Discovering label-specific local models")

    pattern_pool = (
        load_pattern_pool_by_label(
            paths["pattern_pool"]
        )
    )

    (
        local_models_by_label,
        local_summary_df,
    ) = discover_local_models_from_pattern_pool(
        pattern_pool,
        local_models_dir=(
            paths["local_models_dir"]
        ),
        noise_threshold=local_noise,
        progress_every=progress_every,
    )

    local_summary_df.to_csv(
        paths["local_models_summary"],
        index=False,
    )

    print(
        "[Stage05:3] local models:",
        len(local_models_by_label),
    )

    # --------------------------------------------------------
    # 4. Build EXP model
    # --------------------------------------------------------
    print("[Stage05:4] Expanding ABS model with local models")

    (
        expanded_net,
        expanded_initial_marking,
        expanded_final_marking,
        expansion_metadata,
    ) = expand_abstract_petri_net(
        abstract_net,
        abstract_initial_marking,
        abstract_final_marking,
        local_models_by_label,
    )

    expanded_metadata = (
        save_expanded_model(
            expanded_net,
            expanded_initial_marking,
            expanded_final_marking,
            pnml_path=(
                paths["expanded_pnml"]
            ),
            meta_path=(
                paths["expanded_meta"]
            ),
            metadata=(
                expansion_metadata
            ),
        )
    )

    print(
        "[Stage05:4] replaced transitions:",
        expanded_metadata["n_replaced"],
    )
    print(
        "[Stage05:4] missing labels:",
        expanded_metadata["missing_labels"],
    )

    # --------------------------------------------------------
    # 5. Stage report
    # --------------------------------------------------------
    report = {
        "status": "completed",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": cfg["dataset"]["name"],
        "method": cfg["abstraction_source"]["method"],
        "k": int(k),
        "jump": int(jump),
        "rank": int(rank),
        "abstract_noise": float(
            abstract_noise
        ),
        "local_noise": float(
            local_noise
        ),

        "cases": {
            "n_input_cases": int(
                len(case_sequence_map)
            ),
            "n_selected_cases": int(
                len(case_ids)
            ),
            "n_missing_cases": int(
                len(missing_cases)
            ),
            "n_original_prefix_events": int(
                sum(
                    map(
                        len,
                        original_prefix_traces,
                    )
                )
            ),
            "n_abstract_events": int(
                sum(map(len, abstract_traces))
            ),
        },

        "abstract_model": {
            **abstract_tree_outputs,
            **abstract_net_meta,
            "created_this_run": bool(
                abs_created_this_run
            ),
            "reused_existing": bool(
                not abs_created_this_run
            ),
        },

        "local_models": {
            "n_pattern_labels": int(
                len(pattern_pool)
            ),
            "n_successful_models": int(
                len(local_models_by_label)
            ),
            "summary_csv": str(
                paths[
                    "local_models_summary"
                ]
            ),
            "directory": str(
                paths["local_models_dir"]
            ),
        },

        "expanded_model": (
            expanded_metadata
        ),

        "outputs": {
            key: str(value)
            for key, value in paths.items()
        },
    }

    save_json(
        paths["stage_report"],
        report,
    )

    print("=" * 80)
    print("[Stage05] Done")
    print("=" * 80)
    print(
        "[Stage05] ABS model:",
        paths["abstract_pnml"],
    )
    print(
        "[Stage05] EXP model:",
        paths["expanded_pnml"],
    )
    print(
        "[Stage05] report:",
        paths["stage_report"],
    )

    return report


def run_discovery_pipeline(
    cfg: Dict[str, Any],
    *,
    rank: int,
    abstract_noise: float,
    local_noise: float,
    progress_every: int,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []

    for k in cfg["abstraction"]["k_values"]:
        for jump in cfg["abstraction"]["jump_values"]:
            reports.append(
                run_discovery_for_setting(
                    cfg,
                    k=int(k),
                    jump=int(jump),
                    rank=rank,
                    abstract_noise=(
                        abstract_noise
                    ),
                    local_noise=(
                        local_noise
                    ),
                    progress_every=(
                        progress_every
                    ),
                )
            )

    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 05: construct ABS and "
            "expanded EXP process models."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/bpic2015.yaml",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--abstract-noise",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--local-noise",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(
        args.config
    )

    run_discovery_pipeline(
        cfg,
        rank=args.rank,
        abstract_noise=(
            args.abstract_noise
        ),
        local_noise=(
            args.local_noise
        ),
        progress_every=(
            args.progress_every
        ),
    )


if __name__ == "__main__":
    main()