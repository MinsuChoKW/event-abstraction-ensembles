# scripts/04_select_topk_combinations.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from eae.config import load_config
from eae.paths import build_selection_output_paths

from eae.selection.topk_solver import (
    MAX_STORED_COMBINATIONS,
    select_top_combinations_for_cases,
    write_jsonl_gz,
)

from eae.selection.stats import (
    build_case_summary_dataframe,
    build_label_usage_dataframe,
    build_stats_summary_dataframe,
)


def save_json(
    obj: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

    return path


def load_case_lengths_from_summary(
    path: str | Path,
) -> Dict[str, int]:
    """
    Load case length information from matching/dp_case_summary.csv.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"dp_case_summary.csv does not exist: {path}")

    df = pd.read_csv(path)

    if "case_id" not in df.columns:
        raise KeyError(f"'case_id' column is missing in {path}")

    length_col = None

    for candidate in ["case_len", "length", "trace_len"]:
        if candidate in df.columns:
            length_col = candidate
            break

    if length_col is None:
        raise KeyError(
            f"No case length column found in {path}. "
            "Expected one of: case_len, length, trace_len."
        )

    return {
        str(row["case_id"]): int(row[length_col])
        for _, row in df.iterrows()
    }


def run_selection_for_setting(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
) -> Dict[str, Any]:
    """
    Run top combination selection for one K and jump setting.
    """
    output_paths = build_selection_output_paths(
        cfg,
        jump=jump,
        k=k,
    )

    dp_edges_path = output_paths["dp_edges"]
    dp_case_summary_path = output_paths["dp_case_summary"]

    if not dp_edges_path.exists():
        raise FileNotFoundError(f"dp_edges file does not exist: {dp_edges_path}")

    if not dp_case_summary_path.exists():
        raise FileNotFoundError(
            f"dp_case_summary file does not exist: {dp_case_summary_path}"
        )

    requested_top_k = int(cfg["abstraction"].get("top_k_solutions", 5))
    top_k = min(requested_top_k, MAX_STORED_COMBINATIONS)

    print("=" * 80)
    print("[Selection] Start top combination selection")
    print("=" * 80)
    print(f"[Selection] dataset        : {cfg['dataset']['name']}")
    print(f"[Selection] method         : {cfg['abstraction_source']['method']}")
    print(f"[Selection] K              : {k}")
    print(f"[Selection] jump           : {jump}")
    print(f"[Selection] requested_top_k: {requested_top_k}")
    print(f"[Selection] stored_top_k   : {top_k}")
    print(f"[Selection] run_dir        : {output_paths['run_dir']}")
    print(f"[Selection] dp_edges       : {dp_edges_path}")

    # --------------------------------------------------------
    # 1. Load edges and case lengths
    # --------------------------------------------------------
    case_lengths = load_case_lengths_from_summary(dp_case_summary_path)

    print("[Selection] n_cases in summary:", len(case_lengths))
    # --------------------------------------------------------
    # 2. Select top combinations per case
    # --------------------------------------------------------
    solution_rows, case_rows = select_top_combinations_for_cases(
        output_paths["dp_edges"],
        case_lengths=case_lengths,
        top_k=top_k,
        max_delta=50,
        random_seed=42,
        progress_every=10,
    )

    # --------------------------------------------------------
    # 3. Save outputs
    # --------------------------------------------------------
    write_jsonl_gz(
        solution_rows,
        output_paths["selected_combinations"],
    )

    df_case = build_case_summary_dataframe(case_rows)
    df_case.to_csv(output_paths["case_summary"], index=False)

    df_stats = build_stats_summary_dataframe(case_rows)
    df_stats.to_csv(output_paths["stats_summary"], index=False)

    df_usage = build_label_usage_dataframe(
        solution_rows,
        top_rank_only=False,
    )
    df_usage.to_csv(output_paths["label_usage"], index=False)

    report = {
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "dataset": cfg["dataset"]["name"],
        "method": cfg["abstraction_source"]["method"],
        "k": int(k),
        "jump": int(jump),
        "requested_top_k": int(requested_top_k),
        "stored_top_k": int(top_k),
        "outputs": {
            "run_dir": str(output_paths["run_dir"]),
            "dp_edges": str(output_paths["dp_edges"]),
            "dp_case_summary": str(output_paths["dp_case_summary"]),
            "selected_combinations": str(output_paths["selected_combinations"]),
            "case_summary": str(output_paths["case_summary"]),
            "stats_summary": str(output_paths["stats_summary"]),
            "label_usage": str(output_paths["label_usage"]),
            "selection_report": str(output_paths["selection_report"]),
        },
        "summary": {
            "n_edges": int(sum(row.get("n_edges", 0) for row in case_rows)),
            "n_cases": int(len(case_lengths)),
            "n_solution_rows": int(len(solution_rows)),
            "n_case_rows": int(len(case_rows)),
            "n_cases_with_selected_solutions": int(
                sum(1 for row in case_rows if row.get("n_selected_solutions", 0) > 0)
            ),
            "n_cases_reaches_end": int(
                sum(1 for row in case_rows if row.get("reaches_end", False))
            ),
        },
    }

    save_json(report, output_paths["selection_report"])

    print("[Selection] saved selected combinations:", output_paths["selected_combinations"])
    print("[Selection] saved case summary:", output_paths["case_summary"])
    print("[Selection] saved stats summary:", output_paths["stats_summary"])
    print("[Selection] saved label usage:", output_paths["label_usage"])
    print("[Selection] saved report:", output_paths["selection_report"])
    print("=" * 80)
    print("[Selection] Done")
    print("=" * 80)

    return report


def run_selection_pipeline(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run selection for all K and jump values in config.
    """
    k_values = [int(k) for k in cfg["abstraction"]["k_values"]]
    jump_values = [int(j) for j in cfg["abstraction"]["jump_values"]]

    reports = []

    for k in k_values:
        for jump in jump_values:
            report = run_selection_for_setting(
                cfg,
                k=k,
                jump=jump,
            )
            reports.append(report)

    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top decomposition combinations from segment matches."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/bpic2015.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_selection_pipeline(cfg)


if __name__ == "__main__":
    main()