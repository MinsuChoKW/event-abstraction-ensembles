# scripts/03_run_segment_matching.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import gzip

import pm4py

from eae.config import load_config
from eae.paths import build_segment_matching_paths

from eae.patterns.tokenization import load_case_strings
from eae.patterns.pool_builder import (
    build_pattern_pool,
    save_json,
    save_pattern_pool_artifacts,
)
from eae.matching.dp_runner import (
    build_segment_matches_for_cases,
    write_jsonl_gz,
)

def ensure_case_strings_from_event_log(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure processed case-string files exist.

    If the following files do not exist, create them from the raw XES event log:

      data/{dataset}/processed/act2ch.json
      data/{dataset}/processed/case_strings.jsonl.gz

    The segment matching stage needs case_strings.jsonl.gz.
    Therefore this function is called at the beginning of script 03.
    """
    from eae.config import resolve_path

    dataset_cfg = cfg["dataset"]

    log_path = resolve_path(cfg, dataset_cfg["raw_log_path"])
    processed_dir = resolve_path(cfg, dataset_cfg["processed_dir"])

    case_col = dataset_cfg["columns"]["case_id"]
    act_col = dataset_cfg["columns"]["activity"]
    time_col = dataset_cfg["columns"]["timestamp"]

    act2ch_path = processed_dir / dataset_cfg["activity_map_file"]
    case_strings_path = processed_dir / dataset_cfg["case_strings_file"]

    processed_dir.mkdir(parents=True, exist_ok=True)

    if case_strings_path.exists() and act2ch_path.exists():
        print("[CaseStrings] Existing processed files found.")
        print(f"[CaseStrings] act2ch       : {act2ch_path}")
        print(f"[CaseStrings] case_strings: {case_strings_path}")
        return {
            "created": False,
            "act2ch_path": str(act2ch_path),
            "case_strings_path": str(case_strings_path),
        }

    print("[CaseStrings] Processed case strings not found. Creating from raw event log.")
    print(f"[CaseStrings] raw log      : {log_path}")
    print(f"[CaseStrings] processed dir: {processed_dir}")

    if not log_path.exists():
        raise FileNotFoundError(f"Event log does not exist: {log_path}")

    log_obj = pm4py.read_xes(str(log_path))

    if isinstance(log_obj, pd.DataFrame):
        df = log_obj.copy()
    else:
        df = pm4py.convert_to_dataframe(log_obj)

    required_cols = [case_col, act_col, time_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in event log: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[case_col, act_col, time_col]].copy()
    df = df.dropna(subset=[case_col, act_col])

    df[case_col] = df[case_col].astype(str)
    df[act_col] = df[act_col].astype(str)

    df = df.sort_values([case_col, time_col], kind="mergesort").reset_index(drop=True)

    activities = sorted(df[act_col].unique().tolist())

    # BPIC2015는 activity name 자체를 sequence token으로 사용한다.
    # act2ch는 재현/확인용 metadata로 저장한다.
    act2ch = {
        act: act
        for act in activities
    }

    with act2ch_path.open("w", encoding="utf-8") as f:
        json.dump(act2ch, f, ensure_ascii=False, indent=2)

    n_cases = 0
    n_events = 0
    lengths = []

    with gzip.open(case_strings_path, "wt", encoding="utf-8") as f:
        for case_id, g in df.groupby(case_col, sort=False):
            seq = g[act_col].astype(str).tolist()

            row = {
                "case_id": str(case_id),
                "sequence": seq,
                "length": len(seq),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_cases += 1
            n_events += len(seq)
            lengths.append(len(seq))

    report = {
        "created": True,
        "raw_log_path": str(log_path),
        "processed_dir": str(processed_dir),
        "act2ch_path": str(act2ch_path),
        "case_strings_path": str(case_strings_path),
        "n_cases": int(n_cases),
        "n_events": int(n_events),
        "n_activities": int(len(activities)),
        "min_case_len": int(min(lengths)) if lengths else 0,
        "max_case_len": int(max(lengths)) if lengths else 0,
        "mean_case_len": float(sum(lengths) / len(lengths)) if lengths else 0.0,
    }

    print("[CaseStrings] Saved act2ch       :", act2ch_path)
    print("[CaseStrings] Saved case_strings:", case_strings_path)
    print("[CaseStrings] n_cases           :", report["n_cases"])
    print("[CaseStrings] n_events          :", report["n_events"])
    print("[CaseStrings] n_activities      :", report["n_activities"])
    print("[CaseStrings] min_case_len      :", report["min_case_len"])
    print("[CaseStrings] max_case_len      :", report["max_case_len"])
    print("[CaseStrings] mean_case_len     :", report["mean_case_len"])

    return report

def save_report(report: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return path


def run_segment_matching_for_setting(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
) -> Dict[str, Any]:
    """
    Run segment matching for one K and one jump setting.
    """
    use_backward_matching = bool(cfg["abstraction"].get("use_backward_matching", True))

    output_paths = build_segment_matching_paths(
        cfg,
        jump=jump,
        k=k,
    )

    print("=" * 80)
    print("[Matching] Start segment matching")
    print("=" * 80)
    print(f"[Matching] dataset               : {cfg['dataset']['name']}")
    print(f"[Matching] method                : {cfg['abstraction_source']['method']}")
    print(f"[Matching] K                     : {k}")
    print(f"[Matching] jump                  : {jump}")
    print(f"[Matching] use_backward_matching : {use_backward_matching}")
    print(f"[Matching] run_dir               : {output_paths['run_dir']}")

    # --------------------------------------------------------
    # 1. Load case strings
    # --------------------------------------------------------
    ensure_case_strings_from_event_log(cfg)
    
    cases = load_case_strings(cfg)

    n_cases = int(cfg["abstraction"].get("n_cases", len(cases)))

    if n_cases > 0:
        cases = cases[:n_cases]

    print("[Matching] loaded cases:", len(cases))

    # --------------------------------------------------------
    # 2. Build unified pattern pool
    # --------------------------------------------------------
    pattern_pool = build_pattern_pool(
        cfg,
        k=k,
    )

    pool_meta = save_pattern_pool_artifacts(
        pattern_pool,
        cfg=cfg,
        k=k,
        jump=jump,
        output_paths=output_paths,
    )

    print("[Matching] saved pattern pool:", output_paths["pattern_pool_by_label"])
    print("[Matching] n_labels:", pool_meta["summary"]["n_labels"])
    print("[Matching] n_paths :", pool_meta["summary"]["n_paths"])

    # --------------------------------------------------------
    # 3. Segment matching
    # --------------------------------------------------------
    edges, case_summaries, edge_label_counts = build_segment_matches_for_cases(
        cases,
        pattern_pool,
        jump=jump,
        use_backward_matching=use_backward_matching,
    )

    # --------------------------------------------------------
    # 4. Save outputs
    # --------------------------------------------------------
    write_jsonl_gz(edges, output_paths["dp_edges"])

    pd.DataFrame(case_summaries).to_csv(
        output_paths["dp_case_summary"],
        index=False,
    )

    save_json(
        output_paths["edge_label_counts"],
        edge_label_counts,
    )

    report = {
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "dataset": cfg["dataset"]["name"],
        "method": cfg["abstraction_source"]["method"],
        "k": int(k),
        "jump": int(jump),
        "use_backward_matching": use_backward_matching,
        "outputs": {
            "run_dir": str(output_paths["run_dir"]),
            "pattern_pool_by_label": str(output_paths["pattern_pool_by_label"]),
            "pattern_pool_meta": str(output_paths["pattern_pool_meta"]),
            "edge_label_counts": str(output_paths["edge_label_counts"]),
            "dp_edges": str(output_paths["dp_edges"]),
            "dp_case_summary": str(output_paths["dp_case_summary"]),
            "matching_report": str(output_paths["matching_report"]),
        },
        "summary": {
            "n_cases": len(cases),
            "n_edges": len(edges),
            "n_case_summaries": len(case_summaries),
            "n_edge_labels": len(edge_label_counts),
            "n_pattern_labels": pool_meta["summary"]["n_labels"],
            "n_pattern_paths": pool_meta["summary"]["n_paths"],
        },
    }

    save_report(report, output_paths["matching_report"])

    print("[Matching] saved dp_edges:", output_paths["dp_edges"])
    print("[Matching] saved dp_case_summary:", output_paths["dp_case_summary"])
    print("[Matching] saved report:", output_paths["matching_report"])
    print("[Matching] n_edges:", len(edges))
    print("=" * 80)
    print("[Matching] Done")
    print("=" * 80)

    return report


def run_segment_matching_pipeline(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run segment matching for every K and jump setting in config.
    """
    k_values = [int(k) for k in cfg["abstraction"]["k_values"]]
    jump_values = [int(j) for j in cfg["abstraction"]["jump_values"]]

    reports = []

    for k in k_values:
        for jump in jump_values:
            report = run_segment_matching_for_setting(
                cfg,
                k=k,
                jump=jump,
            )
            reports.append(report)

    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run segment matching for event abstraction."
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
    run_segment_matching_pipeline(cfg)


if __name__ == "__main__":
    main()