# scripts/01_prepare_lpm_paths.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from eae.config import load_config, resolve_path

from eae.lpm.groups import (
    get_lpm_groups_from_config,
    analyze_lpm_group_overlaps,
    deduplicate_lpm_group_ids,
    collect_group_ptml_paths,
    summarize_group_paths,
)

from eae.lpm.convert import (
    convert_pnml_dir_to_ptml,
    rename_ptml_files_to_canonical,
)

from eae.lpm.extract_paths import (
    build_lpm_groups_payload,
    save_lpm_groups_payload,
    summarize_lpm_groups_payload,
)

from eae.lpm.deduplicate import deduplicate_lpm_payload_file


def build_lpm_output_paths(
    cfg: Dict[str, Any],
) -> Dict[str, Path]:
    """
    Build LPM-related output paths from config.

    Important
    ---------
    model_based.path_dir is already dataset-specific.

    Example
    -------
    model_based.path_dir:
      data/bpic2015/interim/LPM_Path

    outputs:
      data/bpic2015/interim/LPM_Path/lpm_group_paths_loop2.json
      data/bpic2015/interim/LPM_Path/lpm_group_paths_loop2_remove_dup.json
      data/bpic2015/interim/LPM_Path/lpm_pipeline_report_loop2.json
    """
    model_cfg = cfg["model_based"]

    path_dir = resolve_path(cfg, model_cfg["path_dir"])
    path_dir.mkdir(parents=True, exist_ok=True)

    loop_max_iters = int(model_cfg["loop_constraint"]["loop_max_iters"])

    raw_template = model_cfg["path_output"]["raw_template"]
    dedup_template = model_cfg["path_output"]["dedup_template"]

    raw_json = path_dir / raw_template.format(loop_max_iters=loop_max_iters)
    dedup_json = path_dir / dedup_template.format(loop_max_iters=loop_max_iters)
    report_json = path_dir / f"lpm_pipeline_report_loop{loop_max_iters}.json"

    return {
        "path_dir": path_dir,
        "raw_json": raw_json,
        "dedup_json": dedup_json,
        "report_json": report_json,
    }


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Save LPM pipeline report as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def run_lpm_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full LPM preprocessing pipeline.

    Dataset-specific generated outputs
    ----------------------------------
    PTML files:
      model_based.ptml_dir / dataset.name

    LPM path files:
      model_based.path_dir / dataset.name

    Steps
    -----
    1. Read user-defined LPM groups from config.
    2. Analyze duplicated LPM ids across groups.
    3. Convert PNML files to PTML files.
    4. Rename PTML files to canonical names.
    5. Deduplicate LPM ids by group rank.
    6. Collect PTML paths by group.
    7. Extract bounded executable paths from process trees.
    8. Save raw LPM path payload.
    9. Optionally save path-deduplicated payload.
    """
    model_cfg = cfg["model_based"]
    dataset_name = cfg["dataset"]["name"]

    if not bool(model_cfg.get("enabled", True)):
        print("[LPM] model_based.enabled is false. Skipping LPM pipeline.")
        return {
            "status": "skipped",
            "reason": "model_based.enabled is false",
        }

    # --------------------------------------------------------
    # LPM folders
    # --------------------------------------------------------
    # Important:
    # Config paths are already dataset-specific.
    #
    # Example:
    #   raw_lpm_dir = data/bpic2015/raw/LPM_PN_Set
    #   ptml_dir    = data/bpic2015/interim/LPM_PT_ver
    #   path_dir    = data/bpic2015/interim/LPM_Path
    # --------------------------------------------------------
    raw_lpm_dir = resolve_path(cfg, model_cfg["raw_lpm_dir"])
    ptml_dir = resolve_path(cfg, model_cfg["ptml_dir"])
    path_dir = resolve_path(cfg, model_cfg["path_dir"])

    ptml_dir.mkdir(parents=True, exist_ok=True)
    path_dir.mkdir(parents=True, exist_ok=True)

    loop_cfg = model_cfg.get("loop_constraint", {})
    loop_max_iters = int(loop_cfg.get("loop_max_iters", 2))

    deduplicate_paths = bool(model_cfg.get("deduplicate_paths", True))

    enum_cfg = model_cfg.get("path_enumeration", {})
    max_traces_per_lpm = int(enum_cfg.get("max_traces_per_lpm", 50000))
    max_trace_len = int(enum_cfg.get("max_trace_len", 200))

    out_paths = build_lpm_output_paths(cfg)

    raw_json = out_paths["raw_json"]
    dedup_json = out_paths["dedup_json"]
    report_json = out_paths["report_json"]

    print("=" * 80)
    print("[LPM] Start LPM preprocessing")
    print("=" * 80)
    print(f"[LPM] dataset           : {dataset_name}")
    print(f"[LPM] raw_lpm_dir       : {raw_lpm_dir}")
    print(f"[LPM] ptml_dir          : {ptml_dir}")
    print(f"[LPM] path_dir          : {path_dir}")
    print(f"[LPM] loop_max_iters    : {loop_max_iters}")
    print(f"[LPM] deduplicate_paths : {deduplicate_paths}")
    print(f"[LPM] max_traces_per_lpm: {max_traces_per_lpm}")
    print(f"[LPM] max_trace_len     : {max_trace_len}")
    print(f"[LPM] raw_json          : {raw_json}")
    print(f"[LPM] dedup_json        : {dedup_json}")

    # --------------------------------------------------------
    # 1. Load user-defined LPM groups from config
    # --------------------------------------------------------
    group_ids = get_lpm_groups_from_config(cfg)

    # --------------------------------------------------------
    # 2. Analyze duplicated LPM ids before group-id dedup
    # --------------------------------------------------------
    overlap_info = analyze_lpm_group_overlaps(group_ids)

    print("[LPM] n_groups:", overlap_info["n_groups"])
    print("[LPM] n_unique_lpms:", overlap_info["n_unique_lpms"])
    print("[LPM] duplicated LPM ids:", len(overlap_info["duplicated_lpms"]))

    # --------------------------------------------------------
    # 3. Convert PNML -> PTML
    # --------------------------------------------------------
    convert_report = convert_pnml_dir_to_ptml(
        input_dir=raw_lpm_dir,
        output_dir=ptml_dir,
        overwrite=True,
    )

    print("[LPM] PNML -> PTML converted:", convert_report["n_converted"])
    print("[LPM] PNML -> PTML skipped  :", convert_report["n_skipped"])
    print("[LPM] PNML -> PTML errors   :", convert_report["n_errors"])

    # --------------------------------------------------------
    # 4. Rename PTML files to canonical lpm.<id>.ptml
    # --------------------------------------------------------
    rename_report = rename_ptml_files_to_canonical(
        ptml_dir=ptml_dir,
        dry_run=False,
    )

    print("[LPM] PTML renamed  :", rename_report["n_renamed"])
    print("[LPM] PTML conflicts:", rename_report["n_conflicts"])
    print("[LPM] PTML skipped  :", rename_report["n_skipped"])

    # --------------------------------------------------------
    # 5. Deduplicate LPM ids across groups
    # --------------------------------------------------------
    dedup_group_ids, removed_lpm_report = deduplicate_lpm_group_ids(group_ids)

    print("[LPM] duplicated LPM ids removed from later groups:", len(removed_lpm_report))

    # --------------------------------------------------------
    # 6. Collect group PTML paths
    # --------------------------------------------------------
    groups_paths, missing_report = collect_group_ptml_paths(
        ptml_dir=ptml_dir,
        group_ids=dedup_group_ids,
        strict=False,
    )

    group_path_summary = summarize_group_paths(groups_paths)

    print("[LPM] collected PTML files:", group_path_summary["n_total_ptml"])
    print("[LPM] empty groups:", group_path_summary["empty_groups"])
    print("[LPM] missing PTML:", len(missing_report))

    # --------------------------------------------------------
    # 7. Extract executable paths from process trees
    # --------------------------------------------------------
    payload, build_errors = build_lpm_groups_payload(
        groups_paths=groups_paths,
        ptml_dir=ptml_dir,
        loop_max_iters=loop_max_iters,
        max_traces_per_lpm=max_traces_per_lpm,
        max_trace_len=max_trace_len,
    )

    payload_summary = summarize_lpm_groups_payload(payload)

    print("[LPM] extracted LPM count:", payload_summary["n_lpms"])
    print("[LPM] extracted path count:", payload_summary["n_paths"])
    print("[LPM] build errors:", len(build_errors))

    # --------------------------------------------------------
    # 8. Save raw LPM path payload
    # --------------------------------------------------------
    save_lpm_groups_payload(payload, raw_json)
    print("[LPM] saved raw path payload:", raw_json)

    # --------------------------------------------------------
    # 9. Optionally deduplicate paths globally
    # --------------------------------------------------------
    dedup_report = None

    if deduplicate_paths:
        dedup_report = deduplicate_lpm_payload_file(
            input_json=raw_json,
            output_json=dedup_json,
        )

        print("[LPM] saved dedup path payload:", dedup_json)
        print(
            "[LPM] final unique paths:",
            dedup_report["final_meta"].get("global_unique_paths"),
        )
    else:
        print("[LPM] deduplicate_paths is false. Skipping path deduplication.")

    # --------------------------------------------------------
    # 10. Save pipeline report
    # --------------------------------------------------------
    report = {
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "config_summary": {
            "dataset": dataset_name,
            "raw_lpm_dir": str(raw_lpm_dir),
            "ptml_dir": str(ptml_dir),
            "path_dir": str(path_dir),
            "loop_max_iters": loop_max_iters,
            "deduplicate_paths": deduplicate_paths,
            "max_traces_per_lpm": max_traces_per_lpm,
            "max_trace_len": max_trace_len,
        },
        "outputs": {
            "raw_json": str(raw_json),
            "dedup_json": str(dedup_json) if deduplicate_paths else None,
            "report_json": str(report_json),
        },
        "group_overlap_info": overlap_info,
        "removed_lpm_report": removed_lpm_report,
        "missing_ptml_report": missing_report,
        "group_path_summary": group_path_summary,
        "convert_report": {
            "input_dir": convert_report["input_dir"],
            "output_dir": convert_report["output_dir"],
            "n_converted": convert_report["n_converted"],
            "n_skipped": convert_report["n_skipped"],
            "n_errors": convert_report["n_errors"],
            "errors": convert_report["errors"],
        },
        "rename_report": {
            "ptml_dir": rename_report["ptml_dir"],
            "dry_run": rename_report["dry_run"],
            "n_renamed": rename_report["n_renamed"],
            "n_conflicts": rename_report["n_conflicts"],
            "n_skipped": rename_report["n_skipped"],
            "conflicts": rename_report["conflicts"],
            "skipped": rename_report["skipped"],
        },
        "payload_summary": payload_summary,
        "build_errors": build_errors,
        "dedup_report": dedup_report,
    }

    save_report(report, report_json)

    print("[LPM] saved report:", report_json)
    print("=" * 80)
    print("[LPM] Done")
    print("=" * 80)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LPM paths from raw LPM PNML files."
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
    run_lpm_pipeline(cfg)


if __name__ == "__main__":
    main()