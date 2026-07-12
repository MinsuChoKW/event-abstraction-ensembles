# scripts/06_evaluate_models.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from eae.config import load_config
from eae.paths import (
    build_case_strings_path,
    build_run_dir,
    create_run_subdirs,
)

from eae.discovery.log_builder import (
    build_pm4py_log_from_traces,
    filter_empty,
    load_case_strings_jsonl_gz,
)

from eae.discovery.model_discovery_algo import (
    discover_stable_model,
    save_petri_net,
    save_process_tree,
)

from eae.evaluation.conformance import (
    evaluate_precision_fitness_f1,
)

from eae.evaluation.model_io import (
    load_petri_net,
    load_xes_log,
    summarize_event_log,
    summarize_petri_net,
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


def can_reuse_org_model(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
    *,
    noise_threshold: float,
    n_cases: int,
) -> tuple[bool, str]:
    """
    Existing ORG model is always reused when its discovery
    conditions match the current original-log configuration.
    """
    required_files = [
        paths["org_pnml"],
        paths["org_tree_ptml"],
        paths["org_meta"],
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
            paths["org_meta"]
        )
    except Exception as exc:
        return (
            False,
            f"failed to read ORG metadata: {exc}",
        )

    expected = {
        "model_type": "ORG",
        "dataset": str(
            cfg["dataset"]["name"]
        ),
        "n_cases": int(n_cases),
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
        noise_threshold,
    ):
        return (
            False,
            "org-noise mismatch: "
            f"saved={saved_noise}, "
            f"requested={noise_threshold}",
        )

    return (
        True,
        "all ORG discovery conditions match",
    )

def build_evaluation_paths(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int,
) -> Dict[str, Path]:
    """
    Build stage-06 paths.

    Stage-05 inputs
    ----------------
    discovery/abstracted_log_rank{rank}.xes
    discovery/original_prefix_log_rank{rank}.xes
    discovery/abstract_model_rank{rank}.pnml
    discovery/expanded_model_rank{rank}.pnml

    Stage-06 outputs
    -----------------
    evaluation/conformance_summary_rank{rank}.csv
    evaluation/conformance_details_rank{rank}.json
    evaluation/stage06_report_rank{rank}.json
    """
    run_dir = build_run_dir(
        cfg,
        jump=jump,
        k=k,
    )

    create_run_subdirs(
        cfg,
        run_dir,
    )

    discovery_dir = (
        run_dir / "discovery"
    )

    evaluation_dir = (
        run_dir / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Dataset-level ORG baseline.
    results_root = Path(
        cfg["results"]["root_dir"]
    )

    org_dir = (
        results_root
        / "org_baseline"
        / cfg["dataset"]["name"]
    )

    org_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    org_evaluation_dir = (
        org_dir / "evaluation"
    )

    org_evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "run_dir": run_dir,
        "discovery_dir": discovery_dir,
        "evaluation_dir": evaluation_dir,

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

        "abstract_pnml": (
            discovery_dir
            / f"abstract_model_rank{rank}.pnml"
        ),

        "expanded_pnml": (
            discovery_dir
            / f"expanded_model_rank{rank}.pnml"
        ),

        "org_dir": org_dir,

        "org_tree_text": (
            org_dir / "ORG_tree.tree.txt"
        ),

        "org_tree_ptml": (
            org_dir / "ORG_tree.ptml"
        ),

        "org_pnml": (
            org_dir / "ORG_petri.pnml"
        ),

        "org_meta": (
            org_dir
            / "ORG_petri.net_meta.json"
        ),

        "org_evaluation_dir": org_evaluation_dir,

        "org_conformance_summary": (
            org_evaluation_dir
            / f"org_conformance_summary_rank{rank}.csv"
        ),

        "org_conformance_details": (
            org_evaluation_dir
            / f"org_conformance_details_rank{rank}.json"
        ),

        "summary_csv": (
            evaluation_dir
            / f"conformance_summary_rank{rank}.csv"
        ),

        "details_json": (
            evaluation_dir
            / f"conformance_details_rank{rank}.json"
        ),

        "stage_report": (
            evaluation_dir
            / f"stage06_report_rank{rank}.json"
        ),
    }


def build_original_full_log(
    cfg: Dict[str, Any],
    case_strings_path: str | Path,
):
    """
    Build the fixed original-log sample exactly as in the notebook.

    The first abstraction.n_cases cases are used.
    """
    all_case_sequences = (
        load_case_strings_jsonl_gz(
            case_strings_path
        )
    )

    target_n_cases = int(
        cfg["abstraction"].get(
            "n_cases",
            len(all_case_sequences),
        )
    )

    sample_case_ids = list(
        all_case_sequences.keys()
    )[:target_n_cases]

    traces = [
        list(all_case_sequences[case_id])
        for case_id in sample_case_ids
    ]

    sample_case_ids, traces = filter_empty(
        sample_case_ids,
        traces,
    )

    log = build_pm4py_log_from_traces(
        traces,
        sample_case_ids,
        stable_sort=True,
    )

    return (
        log,
        sample_case_ids,
        traces,
    )


def load_or_create_org_model(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
    original_log,
    *,
    noise_threshold: float,
):
    """
    Always reuse the ORG model when the existing model matches
    the current dataset, case sample, and noise threshold.
    """
    reusable, reason = can_reuse_org_model(
        cfg,
        paths,
        noise_threshold=noise_threshold,
        n_cases=len(original_log),
    )

    if reusable:
        print(
            "[Stage06:ORG] Matching ORG model already exists."
        )
        print(
            "[Stage06:ORG] Skipping ORG discovery."
        )
        print(
            "[Stage06:ORG] reason:",
            reason,
        )
        print(
            "[Stage06:ORG] PNML:",
            paths["org_pnml"],
        )

        (
            net,
            initial_marking,
            final_marking,
        ) = load_petri_net(
            paths["org_pnml"]
        )

        return (
            net,
            initial_marking,
            final_marking,
            False,
        )

    print(
        "[Stage06:ORG] No reusable ORG model."
    )
    print(
        "[Stage06:ORG] reason:",
        reason,
    )
    print(
        "[Stage06:ORG] Discovering ORG model "
        "from the original log."
    )

    (
        tree,
        net,
        initial_marking,
        final_marking,
    ) = discover_stable_model(
        original_log,
        noise_threshold=noise_threshold,
    )

    save_process_tree(
        tree,
        text_path=paths["org_tree_text"],
        ptml_path=paths["org_tree_ptml"],
    )

    org_meta = save_petri_net(
        net,
        initial_marking,
        final_marking,
        pnml_path=paths["org_pnml"],
    )

    org_meta = {
        **org_meta,
        "model_type": "ORG",
        "dataset": str(
            cfg["dataset"]["name"]
        ),
        "noise_threshold": float(
            noise_threshold
        ),
        "n_cases": int(len(original_log)),
    }

    save_json(
        paths["org_meta"],
        org_meta,
    )

    return (
        net,
        initial_marking,
        final_marking,
        True,
    )


SUMMARY_COLUMNS = [
    "model_type",
    "log_type",
    "jump",
    "K",
    "rank",
    "label",
    "cases",
    "events",
    "precision",
    "recall_fitness",
    "f1",
    "log_fitness",
    "percentage_of_fitting_traces",
    "fitness_method",
    "precision_method",
    "noise_threshold",
    "model_path",
]


def make_summary_frame(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        results
    )

    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    return frame[
        SUMMARY_COLUMNS
    ]


def org_conformance_matches(
    result: Dict[str, Any],
    *,
    original_full_summary: Dict[str, Any],
    org_noise: float,
    org_pnml_path: str | Path,
) -> bool:
    try:
        if str(result.get("model_type")) != "ORG":
            return False

        if int(result.get("cases")) != int(
            original_full_summary["n_cases"]
        ):
            return False

        if int(result.get("events")) != int(
            original_full_summary["n_events"]
        ):
            return False

        if not float_equal(
            result.get("noise_threshold"),
            org_noise,
        ):
            return False

        saved_model_path = str(
            result.get("model_path", "")
        )

        if (
            saved_model_path
            and saved_model_path != str(org_pnml_path)
        ):
            return False

    except (TypeError, ValueError):
        return False

    return True


def load_saved_org_conformance(
    paths: Dict[str, Path],
    *,
    original_full_summary: Dict[str, Any],
    org_noise: float,
) -> Dict[str, Any] | None:
    details_path = paths[
        "org_conformance_details"
    ]

    if details_path.exists():
        try:
            payload = load_json(
                details_path
            )

            result = payload.get(
                "result",
                payload,
            )

            if (
                isinstance(result, dict)
                and org_conformance_matches(
                    result,
                    original_full_summary=(
                        original_full_summary
                    ),
                    org_noise=org_noise,
                    org_pnml_path=paths["org_pnml"],
                )
            ):
                print(
                    "[Stage06:ORG] Reusing saved ORG "
                    "conformance result."
                )
                print(
                    "[Stage06:ORG] details:",
                    details_path,
                )
                return dict(result)

        except Exception as exc:
            print(
                "[Stage06:ORG] Failed to read saved "
                f"ORG conformance: {exc}"
            )

    summary_path = paths[
        "org_conformance_summary"
    ]

    if summary_path.exists():
        try:
            frame = pd.read_csv(
                summary_path
            )

            if "model_type" not in frame.columns:
                return None

            org_rows = frame[
                frame["model_type"].astype(str)
                == "ORG"
            ]

            if org_rows.empty:
                return None

            result = org_rows.iloc[0].to_dict()

            if org_conformance_matches(
                result,
                original_full_summary=(
                    original_full_summary
                ),
                org_noise=org_noise,
                org_pnml_path=paths["org_pnml"],
            ):
                print(
                    "[Stage06:ORG] Reusing saved ORG "
                    "conformance summary."
                )
                print(
                    "[Stage06:ORG] summary:",
                    summary_path,
                )
                return dict(result)

        except Exception as exc:
            print(
                "[Stage06:ORG] Failed to read saved "
                f"ORG conformance summary: {exc}"
            )

    return None


def save_org_conformance(
    paths: Dict[str, Path],
    *,
    cfg: Dict[str, Any],
    rank: int,
    original_full_summary: Dict[str, Any],
    org_noise: float,
    result_org: Dict[str, Any],
) -> None:
    make_summary_frame(
        [result_org]
    ).to_csv(
        paths["org_conformance_summary"],
        index=False,
        encoding="utf-8",
    )

    payload = {
        "status": "completed",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": str(
            cfg["dataset"]["name"]
        ),
        "rank": int(rank),
        "org_noise": float(org_noise),
        "original_full_summary": (
            original_full_summary
        ),
        "model_path": str(
            paths["org_pnml"]
        ),
        "result": result_org,
        "outputs": {
            "summary_csv": str(
                paths["org_conformance_summary"]
            ),
            "details_json": str(
                paths["org_conformance_details"]
            ),
        },
    }

    save_json(
        paths["org_conformance_details"],
        payload,
    )

    print(
        "[Stage06:ORG] Saved ORG conformance summary:",
        paths["org_conformance_summary"],
    )
    print(
        "[Stage06:ORG] Saved ORG conformance details:",
        paths["org_conformance_details"],
    )


def load_or_compute_org_conformance(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
    original_full_log,
    org_net,
    org_initial_marking,
    org_final_marking,
    *,
    rank: int,
    org_noise: float,
    original_full_summary: Dict[str, Any],
    org_conformance_cache: Dict[
        tuple[Any, ...],
        Dict[str, Any],
    ] | None,
) -> Dict[str, Any]:
    org_cache_key = (
        "ORG",
        str(cfg["dataset"]["name"]),
        int(original_full_summary["n_cases"]),
        int(original_full_summary["n_events"]),
        float(org_noise),
        str(paths["org_pnml"]),
    )

    if (
        org_conformance_cache is not None
        and org_cache_key in org_conformance_cache
    ):
        print(
            "[Stage06:ORG] Reusing in-memory ORG "
            "conformance result."
        )
        return dict(
            org_conformance_cache[org_cache_key]
        )

    saved_result = load_saved_org_conformance(
        paths,
        original_full_summary=original_full_summary,
        org_noise=org_noise,
    )

    if saved_result is not None:
        if org_conformance_cache is not None:
            org_conformance_cache[
                org_cache_key
            ] = dict(saved_result)

        return saved_result

    print(
        "[Stage06:ORG] No saved ORG conformance result."
    )
    print(
        "[Stage06:ORG] Computing ORG conformance."
    )

    result_org = evaluate_precision_fitness_f1(
        original_full_log,
        org_net,
        org_initial_marking,
        org_final_marking,
        label=(
            "ORG: model(org) "
            "vs log(org)"
        ),
    )

    result_org.update(
        {
            "model_type": "ORG",
            "log_type": "original_full",
            "jump": None,
            "K": None,
            "rank": None,
            "cases": int(
                original_full_summary[
                    "n_cases"
                ]
            ),
            "events": int(
                original_full_summary[
                    "n_events"
                ]
            ),
            "noise_threshold": float(
                org_noise
            ),
            "model_path": str(
                paths["org_pnml"]
            ),
        }
    )

    save_org_conformance(
        paths,
        cfg=cfg,
        rank=rank,
        original_full_summary=(
            original_full_summary
        ),
        org_noise=org_noise,
        result_org=result_org,
    )

    if org_conformance_cache is not None:
        org_conformance_cache[
            org_cache_key
        ] = dict(result_org)

    return result_org


def build_stage06_aggregate_paths(
    cfg: Dict[str, Any],
    *,
    rank: int,
) -> Dict[str, Path]:
    k_values = [
        int(value)
        for value in cfg["abstraction"]["k_values"]
    ]
    jump_values = [
        int(value)
        for value in cfg["abstraction"]["jump_values"]
    ]

    sample_paths = build_evaluation_paths(
        cfg,
        k=k_values[0],
        jump=jump_values[0],
        rank=rank,
    )

    results_root = Path(
        cfg["results"]["root_dir"]
    )

    dataset_name = str(
        cfg["dataset"]["name"]
    )

    relative_run_dir = (
        sample_paths["run_dir"]
        .relative_to(
            results_root
            / "runs"
            / dataset_name
        )
    )

    # Keep:
    #   {dictionary_variant}/method-{method}/match-{direction}
    aggregate_parts = relative_run_dir.parts[:3]

    aggregate_dir = (
        results_root
        / "summaries"
        / dataset_name
        / Path(*aggregate_parts)
    )

    aggregate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "aggregate_dir": aggregate_dir,
        "summary_csv": (
            aggregate_dir
            / f"conformance_summary_rank{rank}.csv"
        ),
        "details_json": (
            aggregate_dir
            / f"conformance_details_rank{rank}.json"
        ),
    }


def save_stage06_aggregate_summary(
    cfg: Dict[str, Any],
    *,
    rank: int,
    reports: List[Dict[str, Any]],
) -> None:
    if not reports:
        return

    rows: List[Dict[str, Any]] = []

    org_result = reports[0].get(
        "org_conformance",
        {},
    ).get("result")

    if isinstance(org_result, dict):
        rows.append(
            dict(org_result)
        )

    for report in reports:
        result_map = report.get(
            "results",
            {},
        )

        for key in ["ABS", "EXP"]:
            result = result_map.get(key)

            if isinstance(result, dict):
                rows.append(
                    dict(result)
                )

    paths = build_stage06_aggregate_paths(
        cfg,
        rank=rank,
    )

    make_summary_frame(
        rows
    ).to_csv(
        paths["summary_csv"],
        index=False,
        encoding="utf-8",
    )

    save_json(
        paths["details_json"],
        {
            "status": "completed",
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "dataset": str(
                cfg["dataset"]["name"]
            ),
            "method": str(
                cfg["abstraction_source"]["method"]
            ),
            "rank": int(rank),
            "n_rows": int(len(rows)),
            "results": rows,
            "source_stage_reports": [
                str(
                    report["outputs"]["stage_report"]
                )
                for report in reports
                if "outputs" in report
                and "stage_report" in report["outputs"]
            ],
            "outputs": {
                "summary_csv": str(
                    paths["summary_csv"]
                ),
                "details_json": str(
                    paths["details_json"]
                ),
            },
        },
    )

    print("=" * 80)
    print("[Stage06] Aggregate conformance summary")
    print("=" * 80)
    print(
        "[Stage06] summary:",
        paths["summary_csv"],
    )
    print(
        "[Stage06] details:",
        paths["details_json"],
    )
    print("=" * 80)


def run_evaluation_for_setting(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
    org_noise: float = 0.20,
    org_conformance_cache: Dict[
        tuple[Any, ...],
        Dict[str, Any],
    ] | None = None,
) -> Dict[str, Any]:
    """
    Run the original notebook's three conformance evaluations:

      1. ORG model vs original full log
      2. ABS model vs abstract log
      3. EXP model vs original prefix log
    """
    paths = build_evaluation_paths(
        cfg,
        k=k,
        jump=jump,
        rank=rank,
    )

    print("=" * 80)
    print("[Stage06] Start conformance evaluation")
    print("=" * 80)
    print(
        f"[Stage06] dataset  : "
        f"{cfg['dataset']['name']}"
    )
    print(
        f"[Stage06] method   : "
        f"{cfg['abstraction_source']['method']}"
    )
    print(f"[Stage06] K        : {k}")
    print(f"[Stage06] jump     : {jump}")
    print(f"[Stage06] rank     : {rank}")
    print(f"[Stage06] ORG noise: {org_noise}")
    print(
        f"[Stage06] run_dir  : "
        f"{paths['run_dir']}"
    )

    required_inputs = [
        paths["case_strings"],
        paths["abstracted_xes"],
        paths["original_prefix_xes"],
        paths["abstract_pnml"],
        paths["expanded_pnml"],
    ]

    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(
                "Required stage-06 input is missing: "
                f"{required_path}"
            )

    # ========================================================
    # 1. Original full log
    # ========================================================
    print(
        "[Stage06:1] Building original full log"
    )

    (
        original_full_log,
        original_case_ids,
        original_traces,
    ) = build_original_full_log(
        cfg,
        paths["case_strings"],
    )

    original_full_summary = (
        summarize_event_log(
            original_full_log
        )
    )

    print(
        "[Stage06:1] cases :",
        original_full_summary["n_cases"],
    )
    print(
        "[Stage06:1] events:",
        original_full_summary["n_events"],
    )

    # ========================================================
    # 2. Load stage-05 logs and models
    # ========================================================
    print(
        "[Stage06:2] Loading ABS/EXP logs and models"
    )

    abstract_log = load_xes_log(
        paths["abstracted_xes"]
    )

    original_prefix_log = load_xes_log(
        paths["original_prefix_xes"]
    )

    (
        abstract_net,
        abstract_initial_marking,
        abstract_final_marking,
    ) = load_petri_net(
        paths["abstract_pnml"]
    )

    (
        expanded_net,
        expanded_initial_marking,
        expanded_final_marking,
    ) = load_petri_net(
        paths["expanded_pnml"]
    )

    abstract_log_summary = (
        summarize_event_log(
            abstract_log
        )
    )

    original_prefix_summary = (
        summarize_event_log(
            original_prefix_log
        )
    )

    abstract_model_summary = (
        summarize_petri_net(
            abstract_net
        )
    )

    expanded_model_summary = (
        summarize_petri_net(
            expanded_net
        )
    )

    # ========================================================
    # 3. ORG baseline model
    # ========================================================
    print(
        "[Stage06:3] Preparing ORG baseline model"
    )

    (
        org_net,
        org_initial_marking,
        org_final_marking,
        org_created,
    ) = load_or_create_org_model(
        cfg,
        paths,
        original_full_log,
        noise_threshold=org_noise,
    )

    org_model_summary = (
        summarize_petri_net(
            org_net
        )
    )

    # ========================================================
    # 4. Conformance evaluations
    # ========================================================
    print(
        "[Stage06:4] Preparing ORG conformance result"
    )

    result_org = load_or_compute_org_conformance(
        cfg,
        paths,
        original_full_log,
        org_net,
        org_initial_marking,
        org_final_marking,
        rank=rank,
        org_noise=org_noise,
        original_full_summary=original_full_summary,
        org_conformance_cache=org_conformance_cache,
    )

    print(
        "[Stage06:4] Evaluating ABS model "
        "against abstract log"
    )

    result_abs = (
        evaluate_precision_fitness_f1(
            abstract_log,
            abstract_net,
            abstract_initial_marking,
            abstract_final_marking,
            label=(
                f"K{k:03d}: model(abs) "
                "vs log(abs)"
            ),
        )
    )

    result_abs.update(
        {
            "model_type": "ABS",
            "log_type": "abstracted",
            "jump": int(jump),
            "K": int(k),
            "rank": int(rank),
            "cases": int(
                abstract_log_summary[
                    "n_cases"
                ]
            ),
            "events": int(
                abstract_log_summary[
                    "n_events"
                ]
            ),
            "noise_threshold": 0.20,
            "model_path": str(
                paths["abstract_pnml"]
            ),
        }
    )

    print(
        "[Stage06:4] Evaluating EXP model "
        "against original prefix log"
    )

    result_exp = (
        evaluate_precision_fitness_f1(
            original_prefix_log,
            expanded_net,
            expanded_initial_marking,
            expanded_final_marking,
            label=(
                f"K{k:03d}: model(exp) "
                "vs log(orig_prefix)"
            ),
        )
    )

    result_exp.update(
        {
            "model_type": "EXP",
            "log_type": "original_prefix",
            "jump": int(jump),
            "K": int(k),
            "rank": int(rank),
            "cases": int(
                original_prefix_summary[
                    "n_cases"
                ]
            ),
            "events": int(
                original_prefix_summary[
                    "n_events"
                ]
            ),
            "noise_threshold": 0.20,
            "model_path": str(
                paths["expanded_pnml"]
            ),
        }
    )

    # Per-setting outputs store only ABS and EXP.
    # ORG is stored once under results/org_baseline/{dataset}/evaluation/.
    results = [
        result_abs,
        result_exp,
    ]

    # ========================================================
    # 5. Save outputs
    # ========================================================
    summary_columns = SUMMARY_COLUMNS

    summary_df = pd.DataFrame(
        results
    )

    for column in summary_columns:
        if column not in summary_df.columns:
            summary_df[column] = None

    summary_df = summary_df[
        summary_columns
    ]

    summary_df.to_csv(
        paths["summary_csv"],
        index=False,
        encoding="utf-8",
    )

    save_json(
        paths["details_json"],
        results,
    )

    report = {
        "status": "completed",
        "created_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "dataset": cfg["dataset"]["name"],
        "method": (
            cfg[
                "abstraction_source"
            ]["method"]
        ),
        "jump": int(jump),
        "K": int(k),
        "rank": int(rank),

        "evaluation_setting": {
            "fitness": (
                "token_based_replay"
            ),
            "precision": (
                "etconformance_token"
            ),
            "primary_metric": "f1",
        },

        "logs": {
            "original_full": (
                original_full_summary
            ),
            "original_prefix": (
                original_prefix_summary
            ),
            "abstracted": (
                abstract_log_summary
            ),
        },

        "models": {
            "ORG": {
                **org_model_summary,
                "path": str(
                    paths["org_pnml"]
                ),
                "created_this_run": bool(
                    org_created
                ),
            },
            "ABS": {
                **abstract_model_summary,
                "path": str(
                    paths["abstract_pnml"]
                ),
            },
            "EXP": {
                **expanded_model_summary,
                "path": str(
                    paths["expanded_pnml"]
                ),
            },
        },

        "org_conformance": {
            "summary_csv": str(
                paths["org_conformance_summary"]
            ),
            "details_json": str(
                paths["org_conformance_details"]
            ),
            "result": result_org,
        },

        "results": {
            "ABS": result_abs,
            "EXP": result_exp,
        },

        "outputs": {
            "summary_csv": str(
                paths["summary_csv"]
            ),
            "details_json": str(
                paths["details_json"]
            ),
            "stage_report": str(
                paths["stage_report"]
            ),
        },
    }

    save_json(
        paths["stage_report"],
        report,
    )

    print("=" * 80)
    print("[Stage06] Evaluation summary")
    print("=" * 80)

    for result in [result_org] + results:
        print(
            f"[{result['model_type']}] "
            f"precision={result['precision']:.6f} "
            f"| fitness="
            f"{result['recall_fitness']:.6f} "
            f"| F1={result['f1']:.6f}"
        )

    print(
        "[Stage06] summary:",
        paths["summary_csv"],
    )
    print(
        "[Stage06] details:",
        paths["details_json"],
    )
    print(
        "[Stage06] report :",
        paths["stage_report"],
    )
    print("=" * 80)
    print("[Stage06] Done")
    print("=" * 80)

    return report


def run_evaluation_pipeline(
    cfg: Dict[str, Any],
    *,
    rank: int,
    org_noise: float,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []

    # ORG baseline is fixed for a given dataset, sample size,
    # and ORG noise threshold. Therefore, its conformance result
    # can be reused across K/jump settings within this Stage-06 run.
    org_conformance_cache: Dict[
        tuple[Any, ...],
        Dict[str, Any],
    ] = {}

    for k in cfg["abstraction"]["k_values"]:
        for jump in cfg[
            "abstraction"
        ]["jump_values"]:
            reports.append(
                run_evaluation_for_setting(
                    cfg,
                    k=int(k),
                    jump=int(jump),
                    rank=int(rank),
                    org_noise=float(
                        org_noise
                    ),
                    org_conformance_cache=(
                        org_conformance_cache
                    ),
                )
            )

    save_stage06_aggregate_summary(
        cfg,
        rank=rank,
        reports=reports,
    )

    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 06: evaluate ORG, ABS, and "
            "EXP process models using "
            "token-based replay fitness and "
            "ET-conformance precision."
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
        "--org-noise",
        type=float,
        default=0.20,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(
        args.config
    )

    run_evaluation_pipeline(
        cfg,
        rank=args.rank,
        org_noise=args.org_noise,
    )


if __name__ == "__main__":
    main()