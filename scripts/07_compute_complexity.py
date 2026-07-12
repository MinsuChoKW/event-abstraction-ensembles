# scripts/07_compute_complexity.py

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from eae.config import load_config, resolve_path
from eae.paths import (
    build_run_dir,
    create_run_subdirs,
)

from eae.evaluation.complexity import (
    compute_bpmn_complexity_metrics,
    convert_ptml_to_bpmn,
    verify_understandbpmn_ready,
)


COMPLEXITY_COLUMNS = [
    "model_type",
    "dataset",
    "method",
    "jump",
    "K",
    "rank",
    "bpmn_path",
    "coef_network_connectivity",
    "cognitive_weight",
    "connector_heterogeneity",
    "control_flow_complexity",
    "density",
    "sequentiality",
    "token_split",
    "inverse_arc_degree",
]


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


def build_complexity_paths(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int,
) -> Dict[str, Path]:
    """
    Build inputs and outputs for stage 07.

    ORG process tree:
      results/org_baseline/{dataset}/ORG_tree.ptml

    ABS process tree:
      run/discovery/abstract_model_rank{rank}.ptml
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

    complexity_dir = (
        evaluation_dir / "complexity"
    )

    complexity_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_root = resolve_path(
        cfg,
        cfg["results"]["root_dir"],
    )

    org_dir = (
        results_root
        / "org_baseline"
        / cfg["dataset"]["name"]
    )

    org_complexity_dir = (
        org_dir / "complexity"
    )

    org_complexity_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "run_dir": run_dir,
        "discovery_dir": discovery_dir,
        "evaluation_dir": evaluation_dir,
        "complexity_dir": complexity_dir,

        "org_ptml": (
            org_dir / "ORG_tree.ptml"
        ),

        "abs_ptml": (
            discovery_dir
            / f"abstract_model_rank{rank}.ptml"
        ),

        "org_complexity_dir": org_complexity_dir,

        "org_bpmn": (
            org_complexity_dir
            / "ORG_model.bpmn"
        ),

        "org_metrics_csv": (
            org_complexity_dir
            / f"org_complexity_metrics_rank{rank}.csv"
        ),

        "org_metrics_json": (
            org_complexity_dir
            / f"org_complexity_metrics_rank{rank}.json"
        ),

        "abs_bpmn": (
            complexity_dir
            / f"ABS_model_rank{rank}.bpmn"
        ),

        "metrics_csv": (
            complexity_dir
            / f"complexity_metrics_rank{rank}.csv"
        ),

        "metrics_json": (
            complexity_dir
            / f"complexity_metrics_rank{rank}.json"
        ),

        "stage_report": (
            complexity_dir
            / f"stage07_report_rank{rank}.json"
        ),
    }


def install_understandbpmn(
    project_root: str | Path = ".",
) -> None:
    """
    Run the repository installation script when explicitly requested.
    """
    project_root = Path(project_root)

    install_script = (
        project_root
        / "scripts"
        / "install_understandbpmn.sh"
    )

    if not install_script.exists():
        raise FileNotFoundError(
            f"Installation script does not exist: {install_script}"
        )

    print(
        "[Stage07] Installing R and understandBPMN..."
    )

    subprocess.run(
        [
            "bash",
            str(install_script),
        ],
        check=True,
    )


def compute_model_row(
    *,
    model_type: str,
    bpmn_path: Path,
    cfg: Dict[str, Any],
    k: int | None,
    jump: int | None,
    rank: int | None,
) -> Dict[str, Any]:
    print("-" * 80)
    print(
        f"[Stage07] Computing complexity: {model_type}"
    )
    print(
        f"[Stage07] BPMN: {bpmn_path}"
    )

    metrics = compute_bpmn_complexity_metrics(
        bpmn_path
    )

    row = {
        "model_type": str(model_type),
        "dataset": cfg["dataset"]["name"],
        "method": (
            "ORG"
            if model_type == "ORG"
            else cfg[
                "abstraction_source"
            ]["method"]
        ),
        "jump": jump,
        "K": k,
        "rank": rank,
        **metrics,
    }

    print(
        "[Stage07] coefficient_network_connectivity:",
        row["coef_network_connectivity"],
    )
    print(
        "[Stage07] cognitive_weight:",
        row["cognitive_weight"],
    )
    print(
        "[Stage07] connector_heterogeneity:",
        row["connector_heterogeneity"],
    )
    print(
        "[Stage07] control_flow_complexity:",
        row["control_flow_complexity"],
    )
    print(
        "[Stage07] density:",
        row["density"],
    )
    print(
        "[Stage07] sequentiality:",
        row["sequentiality"],
    )
    print(
        "[Stage07] token_split:",
        row["token_split"],
    )
    print(
        "[Stage07] inverse_arc_degree:",
        row["inverse_arc_degree"],
    )

    return row



def make_complexity_frame(
    rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        rows
    )

    for column in COMPLEXITY_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None

    return dataframe[
        COMPLEXITY_COLUMNS
    ]


def org_complexity_matches(
    row: Dict[str, Any],
    *,
    cfg: Dict[str, Any],
    org_bpmn_path: str | Path,
) -> bool:
    if str(row.get("model_type")) != "ORG":
        return False

    if str(row.get("dataset")) != str(
        cfg["dataset"]["name"]
    ):
        return False

    saved_bpmn_path = str(
        row.get("bpmn_path", "")
    )

    if (
        saved_bpmn_path
        and saved_bpmn_path != str(org_bpmn_path)
    ):
        return False

    return True


def load_saved_org_complexity(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> Dict[str, Any] | None:
    json_path = paths["org_metrics_json"]

    if json_path.exists():
        try:
            payload = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )

            row = payload.get(
                "result",
                payload,
            )

            if (
                isinstance(row, dict)
                and org_complexity_matches(
                    row,
                    cfg=cfg,
                    org_bpmn_path=paths["org_bpmn"],
                )
            ):
                print(
                    "[Stage07:ORG] Reusing saved ORG "
                    "complexity metrics."
                )
                print(
                    "[Stage07:ORG] metrics:",
                    json_path,
                )
                return dict(row)

        except Exception as exc:
            print(
                "[Stage07:ORG] Failed to read saved "
                f"ORG complexity JSON: {exc}"
            )

    csv_path = paths["org_metrics_csv"]

    if csv_path.exists():
        try:
            frame = pd.read_csv(
                csv_path
            )

            if "model_type" not in frame.columns:
                return None

            org_rows = frame[
                frame["model_type"].astype(str)
                == "ORG"
            ]

            if org_rows.empty:
                return None

            row = org_rows.iloc[0].to_dict()

            if org_complexity_matches(
                row,
                cfg=cfg,
                org_bpmn_path=paths["org_bpmn"],
            ):
                print(
                    "[Stage07:ORG] Reusing saved ORG "
                    "complexity summary."
                )
                print(
                    "[Stage07:ORG] summary:",
                    csv_path,
                )
                return dict(row)

        except Exception as exc:
            print(
                "[Stage07:ORG] Failed to read saved "
                f"ORG complexity CSV: {exc}"
            )

    return None


def save_org_complexity(
    paths: Dict[str, Path],
    *,
    cfg: Dict[str, Any],
    rank: int,
    org_row: Dict[str, Any],
) -> None:
    make_complexity_frame(
        [org_row]
    ).to_csv(
        paths["org_metrics_csv"],
        index=False,
        encoding="utf-8-sig",
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
        "result": org_row,
        "outputs": {
            "org_bpmn": str(
                paths["org_bpmn"]
            ),
            "metrics_csv": str(
                paths["org_metrics_csv"]
            ),
            "metrics_json": str(
                paths["org_metrics_json"]
            ),
        },
    }

    save_json(
        paths["org_metrics_json"],
        payload,
    )

    print(
        "[Stage07:ORG] Saved ORG complexity CSV :",
        paths["org_metrics_csv"],
    )
    print(
        "[Stage07:ORG] Saved ORG complexity JSON:",
        paths["org_metrics_json"],
    )


def load_or_compute_org_complexity(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
    *,
    rank: int,
    org_complexity_cache: Dict[
        str,
        Dict[str, Any],
    ] | None,
) -> Dict[str, Any]:
    cache_key = str(
        paths["org_bpmn"]
    )

    if (
        org_complexity_cache is not None
        and cache_key in org_complexity_cache
    ):
        print(
            "[Stage07:ORG] Reusing in-memory ORG "
            "complexity metrics."
        )
        return dict(
            org_complexity_cache[cache_key]
        )

    saved_row = load_saved_org_complexity(
        cfg,
        paths,
    )

    if saved_row is not None:
        if org_complexity_cache is not None:
            org_complexity_cache[
                cache_key
            ] = dict(saved_row)

        return saved_row

    print(
        "[Stage07:ORG] No saved ORG complexity metrics."
    )

    if not paths["org_bpmn"].exists():
        print(
            "[Stage07:ORG] Converting ORG PTML to BPMN"
        )

        convert_ptml_to_bpmn(
            paths["org_ptml"],
            paths["org_bpmn"],
        )
    else:
        print(
            "[Stage07:ORG] Reusing existing ORG BPMN:",
            paths["org_bpmn"],
        )

    print(
        "[Stage07:ORG] Computing ORG metrics"
    )

    org_row = compute_model_row(
        model_type="ORG",
        bpmn_path=paths["org_bpmn"],
        cfg=cfg,
        k=None,
        jump=None,
        rank=None,
    )

    save_org_complexity(
        paths,
        cfg=cfg,
        rank=rank,
        org_row=org_row,
    )

    if org_complexity_cache is not None:
        org_complexity_cache[
            cache_key
        ] = dict(org_row)

    return org_row


def build_stage07_aggregate_paths(
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

    sample_paths = build_complexity_paths(
        cfg,
        k=k_values[0],
        jump=jump_values[0],
        rank=rank,
    )

    results_root = resolve_path(
        cfg,
        cfg["results"]["root_dir"],
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
        "metrics_csv": (
            aggregate_dir
            / f"complexity_metrics_rank{rank}.csv"
        ),
        "metrics_json": (
            aggregate_dir
            / f"complexity_metrics_rank{rank}.json"
        ),
    }


def save_stage07_aggregate_summary(
    cfg: Dict[str, Any],
    *,
    rank: int,
    reports: List[Dict[str, Any]],
) -> None:
    if not reports:
        return

    rows: List[Dict[str, Any]] = []

    org_result = reports[0].get(
        "org_complexity",
        {},
    ).get("result")

    if isinstance(org_result, dict):
        rows.append(
            dict(org_result)
        )

    for report in reports:
        abs_result = report.get(
            "results",
            {},
        ).get("ABS")

        if isinstance(abs_result, dict):
            rows.append(
                dict(abs_result)
            )

    paths = build_stage07_aggregate_paths(
        cfg,
        rank=rank,
    )

    make_complexity_frame(
        rows
    ).to_csv(
        paths["metrics_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        paths["metrics_json"],
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
                "metrics_csv": str(
                    paths["metrics_csv"]
                ),
                "metrics_json": str(
                    paths["metrics_json"]
                ),
            },
        },
    )

    print("=" * 80)
    print("[Stage07] Aggregate complexity summary")
    print("=" * 80)
    print(
        "[Stage07] metrics CSV :",
        paths["metrics_csv"],
    )
    print(
        "[Stage07] metrics JSON:",
        paths["metrics_json"],
    )
    print("=" * 80)


def run_complexity_for_setting(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
    install_dependencies: bool = False,
    org_complexity_cache: Dict[
        str,
        Dict[str, Any],
    ] | None = None,
) -> Dict[str, Any]:
    paths = build_complexity_paths(
        cfg,
        k=k,
        jump=jump,
        rank=rank,
    )

    print("=" * 80)
    print("[Stage07] Start complexity evaluation")
    print("=" * 80)
    print(
        f"[Stage07] dataset: "
        f"{cfg['dataset']['name']}"
    )
    print(
        f"[Stage07] method : "
        f"{cfg['abstraction_source']['method']}"
    )
    print(f"[Stage07] K      : {k}")
    print(f"[Stage07] jump   : {jump}")
    print(f"[Stage07] rank   : {rank}")
    print(
        f"[Stage07] run_dir: "
        f"{paths['run_dir']}"
    )

    if install_dependencies:
        install_understandbpmn(
            cfg.get(
                "project",
                {},
            ).get(
                "root_dir",
                ".",
            )
        )

    verify_understandbpmn_ready()

    required_ptml_files = [
        paths["org_ptml"],
        paths["abs_ptml"],
    ]

    for ptml_path in required_ptml_files:
        if not ptml_path.exists():
            raise FileNotFoundError(
                "Required process-tree file is missing: "
                f"{ptml_path}\n"
                "Run stages 05 and 06 first."
            )

    # --------------------------------------------------------
    # 1. ORG is dataset-level. Load or compute once.
    # --------------------------------------------------------
    print("[Stage07:1] Preparing ORG complexity result")

    org_row = load_or_compute_org_complexity(
        cfg,
        paths,
        rank=rank,
        org_complexity_cache=org_complexity_cache,
    )

    # --------------------------------------------------------
    # 2. ABS is setting-specific. Convert and compute every time.
    # --------------------------------------------------------
    print("[Stage07:2] Converting ABS PTML to BPMN")

    convert_ptml_to_bpmn(
        paths["abs_ptml"],
        paths["abs_bpmn"],
    )

    print("[Stage07:2] Computing ABS metrics")

    abs_row = compute_model_row(
        model_type="ABS",
        bpmn_path=paths["abs_bpmn"],
        cfg=cfg,
        k=int(k),
        jump=int(jump),
        rank=int(rank),
    )

    # Per-setting outputs store only ABS.
    # ORG is stored once under results/org_baseline/{dataset}/complexity/.
    rows: List[Dict[str, Any]] = [
        abs_row,
    ]

    # --------------------------------------------------------
    # 3. Save outputs
    # --------------------------------------------------------
    dataframe = make_complexity_frame(
        rows
    )

    dataframe.to_csv(
        paths["metrics_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        paths["metrics_json"],
        rows,
    )

    report = {
        "status": "completed",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": cfg["dataset"]["name"],
        "method": (
            cfg[
                "abstraction_source"
            ]["method"]
        ),
        "K": int(k),
        "jump": int(jump),
        "rank": int(rank),

        "metric_setting": {
            "understandBPMN_version": "1.1.1",
            "bpmn_metrics": [
                "coef_network_connectivity",
                "cognitive_weight",
                "connector_heterogeneity",
                "control_flow_complexity",
                "density",
                "sequentiality",
                "token_split",
            ],
            "petri_net_metric": (
                "inverse_arc_degree"
            ),
        },

        "inputs": {
            "org_ptml": str(
                paths["org_ptml"]
            ),
            "abs_ptml": str(
                paths["abs_ptml"]
            ),
        },

        "org_complexity": {
            "org_bpmn": str(
                paths["org_bpmn"]
            ),
            "metrics_csv": str(
                paths["org_metrics_csv"]
            ),
            "metrics_json": str(
                paths["org_metrics_json"]
            ),
            "result": org_row,
        },

        "outputs": {
            "abs_bpmn": str(
                paths["abs_bpmn"]
            ),
            "metrics_csv": str(
                paths["metrics_csv"]
            ),
            "metrics_json": str(
                paths["metrics_json"]
            ),
            "stage_report": str(
                paths["stage_report"]
            ),
        },

        "results": {
            "ABS": abs_row,
        },
    }

    save_json(
        paths["stage_report"],
        report,
    )

    print("=" * 80)
    print("[Stage07] Complexity summary")
    print("=" * 80)

    display_dataframe = make_complexity_frame(
        [org_row] + rows
    )

    print(
        display_dataframe[
            [
                "model_type",
                "coef_network_connectivity",
                "cognitive_weight",
                "connector_heterogeneity",
                "control_flow_complexity",
                "density",
                "sequentiality",
                "token_split",
                "inverse_arc_degree",
            ]
        ].to_string(index=False)
    )

    print(
        "[Stage07] saved CSV :",
        paths["metrics_csv"],
    )
    print(
        "[Stage07] saved JSON:",
        paths["metrics_json"],
    )
    print(
        "[Stage07] report    :",
        paths["stage_report"],
    )
    print("=" * 80)
    print("[Stage07] Done")
    print("=" * 80)

    return report


def run_complexity_pipeline(
    cfg: Dict[str, Any],
    *,
    rank: int,
    install_dependencies: bool,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []

    org_complexity_cache: Dict[
        str,
        Dict[str, Any],
    ] = {}

    first_run = True

    for k in cfg["abstraction"]["k_values"]:
        for jump in cfg[
            "abstraction"
        ]["jump_values"]:
            reports.append(
                run_complexity_for_setting(
                    cfg,
                    k=int(k),
                    jump=int(jump),
                    rank=int(rank),
                    install_dependencies=(
                        install_dependencies
                        and first_run
                    ),
                    org_complexity_cache=(
                        org_complexity_cache
                    ),
                )
            )

            first_run = False

    save_stage07_aggregate_summary(
        cfg,
        rank=rank,
        reports=reports,
    )

    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 07: compute ORG and ABS "
            "process-model complexity metrics."
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
        "--install-understandbpmn",
        action="store_true",
        help=(
            "Install R and understandBPMN "
            "before computing metrics."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(
        args.config
    )

    run_complexity_pipeline(
        cfg,
        rank=args.rank,
        install_dependencies=(
            args.install_understandbpmn
        ),
    )


if __name__ == "__main__":
    main()