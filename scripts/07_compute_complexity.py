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

        "org_bpmn": (
            complexity_dir
            / "ORG_model.bpmn"
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


def run_complexity_for_setting(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
    install_dependencies: bool = False,
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
    # 1. PTML -> BPMN
    # --------------------------------------------------------
    print("[Stage07:1] Converting ORG PTML to BPMN")

    convert_ptml_to_bpmn(
        paths["org_ptml"],
        paths["org_bpmn"],
    )

    print("[Stage07:1] Converting ABS PTML to BPMN")

    convert_ptml_to_bpmn(
        paths["abs_ptml"],
        paths["abs_bpmn"],
    )

    # --------------------------------------------------------
    # 2. Compute metrics
    # --------------------------------------------------------
    print("[Stage07:2] Computing ORG metrics")

    org_row = compute_model_row(
        model_type="ORG",
        bpmn_path=paths["org_bpmn"],
        cfg=cfg,
        k=None,
        jump=None,
        rank=None,
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

    rows: List[Dict[str, Any]] = [
        org_row,
        abs_row,
    ]

    # --------------------------------------------------------
    # 3. Save outputs
    # --------------------------------------------------------
    dataframe = pd.DataFrame(rows)

    for column in COMPLEXITY_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None

    dataframe = dataframe[
        COMPLEXITY_COLUMNS
    ]

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

        "outputs": {
            "org_bpmn": str(
                paths["org_bpmn"]
            ),
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
            "ORG": org_row,
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

    print(
        dataframe[
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
                )
            )

            first_run = False

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