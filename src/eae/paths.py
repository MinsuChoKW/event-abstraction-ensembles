# src/eae/paths.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def get_project_root(cfg: Dict[str, Any]) -> Path:
    """
    Return project root directory from config.
    """
    return Path(cfg.get("project", {}).get("root_dir", "."))


def resolve_path(cfg: Dict[str, Any], path: str | Path) -> Path:
    """
    Resolve a path relative to project.root_dir unless it is absolute.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return get_project_root(cfg) / path


# ============================================================
# Results/run paths
# ============================================================

def get_dictionary_variant(cfg: Dict[str, Any]) -> str:
    """
    Build simplified dictionary variant name.

    Current naming rule:
      dict_loop2
      dict_loop3

    Dedup/session hyperparameter names are intentionally not included.
    """
    loop_max_iters = int(cfg["model_based"]["loop_constraint"]["loop_max_iters"])

    template = cfg.get("naming", {}).get(
        "dictionary_variant_template",
        "dict_loop{loop_max_iters}",
    )

    return template.format(loop_max_iters=loop_max_iters)


def get_match_direction(cfg: Dict[str, Any]) -> str:
    """
    Return bidir/fwd according to use_backward_matching.
    """
    use_backward = bool(cfg["abstraction"].get("use_backward_matching", True))
    direction_cfg = cfg.get("naming", {}).get("match_direction", {})

    if use_backward:
        return direction_cfg.get("backward_true", "bidir")

    return direction_cfg.get("backward_false", "fwd")


def build_run_dir(
    cfg: Dict[str, Any],
    *,
    jump: int,
    k: int,
) -> Path:
    """
    Build run directory.

    Result structure:
      results/runs/{dataset.name}/dict_loop{n}/method-{method}/match-{bidir|fwd}/jumpXX/KXXX
    """
    results_root = resolve_path(cfg, cfg["results"]["root_dir"])
    template = cfg["results"]["run_dir_template"]

    dataset_name = cfg["dataset"]["name"]
    dictionary_variant = get_dictionary_variant(cfg)
    method = cfg["abstraction_source"]["method"]
    match_direction = get_match_direction(cfg)

    relative_path = template.format(
        dataset_name=dataset_name,
        dictionary_variant=dictionary_variant,
        method=method,
        match_direction=match_direction,
        jump=int(jump),
        K=int(k),
    )

    return results_root / relative_path


def create_run_subdirs(cfg: Dict[str, Any], run_dir: str | Path) -> None:
    """
    Create standard run subdirectories.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for subdir in cfg["results"].get("run_subdirs", []):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)


def build_segment_matching_paths(
    cfg: Dict[str, Any],
    *,
    jump: int,
    k: int,
) -> Dict[str, Path]:
    """
    Build output paths for pattern pool and segment matching.
    """
    run_dir = build_run_dir(cfg, jump=jump, k=k)
    create_run_subdirs(cfg, run_dir)

    pattern_pool_dir = run_dir / "pattern_pool"
    matching_dir = run_dir / "matching"

    pattern_pool_dir.mkdir(parents=True, exist_ok=True)
    matching_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "pattern_pool_dir": pattern_pool_dir,
        "matching_dir": matching_dir,
        "pattern_pool_by_label": pattern_pool_dir / "pattern_pool_by_label.json.gz",
        "pattern_pool_meta": pattern_pool_dir / "pattern_pool_meta.json",
        "edge_label_counts": pattern_pool_dir / "edge_label_counts.json",
        "dp_edges": matching_dir / "dp_edges.jsonl.gz",
        "dp_case_summary": matching_dir / "dp_case_summary.csv",
        "matching_report": matching_dir / "segment_matching_report.json",
    }


def build_selection_output_paths(
    cfg: Dict[str, Any],
    *,
    jump: int,
    k: int,
) -> Dict[str, Path]:
    """
    Build output paths for top combination selection.
    """
    run_dir = build_run_dir(cfg, jump=jump, k=k)
    create_run_subdirs(cfg, run_dir)

    matching_dir = run_dir / "matching"
    selection_dir = run_dir / "selection"

    selection_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "matching_dir": matching_dir,
        "selection_dir": selection_dir,

        "dp_edges": matching_dir / "dp_edges.jsonl.gz",
        "dp_case_summary": matching_dir / "dp_case_summary.csv",

        "selected_combinations": selection_dir / "selected_combinations_top5.jsonl.gz",
        "case_summary": selection_dir / "combination_case_summary.csv",
        "stats_summary": selection_dir / "combination_stats_summary.csv",
        "label_usage": selection_dir / "combination_label_usage.csv",
        "selection_report": selection_dir / "selection_report.json",
    }


# ============================================================
# Dataset/data paths
# ============================================================

def build_processed_dataset_dir(cfg: Dict[str, Any]) -> Path:
    """
    Build processed data directory.

    Important
    ---------
    dataset.processed_dir is already dataset-specific.

    Example:
      data/bpic2015/processed
    """
    return resolve_path(cfg, cfg["dataset"]["processed_dir"])


def build_case_strings_path(cfg: Dict[str, Any]) -> Path:
    """
    Build case strings path.

    Example:
      data/bpic2015/processed/case_strings.jsonl.gz

    If case_strings_template is defined:
      data/bpic2015/processed/case_strings_N832.jsonl.gz
    """
    processed_dir = build_processed_dataset_dir(cfg)
    n_cases = int(cfg["abstraction"]["n_cases"])

    if "case_strings_template" in cfg["dataset"]:
        filename = cfg["dataset"]["case_strings_template"].format(n_cases=n_cases)
    else:
        filename = cfg["dataset"].get("case_strings_file", "case_strings.jsonl.gz")

    return processed_dir / filename


def build_activity_map_path(cfg: Dict[str, Any]) -> Path:
    """
    Build activity map path.

    Example:
      data/bpic2015/processed/act2ch.json
    """
    processed_dir = build_processed_dataset_dir(cfg)
    filename = cfg["dataset"]["activity_map_file"]

    return processed_dir / filename


def build_lpm_path_file(cfg: Dict[str, Any]) -> Path:
    """
    Build LPM path file.

    Important
    ---------
    model_based.path_dir is already dataset-specific.

    Example:
      data/bpic2015/interim/LPM_Path/lpm_group_paths_loop2_remove_dup.json
    """
    model_cfg = cfg["model_based"]

    path_dir = resolve_path(cfg, model_cfg["path_dir"])
    loop_max_iters = int(model_cfg["loop_constraint"]["loop_max_iters"])

    # Dedup is fixed/default in the current project setting.
    template = model_cfg["path_output"]["dedup_template"]

    return path_dir / template.format(loop_max_iters=loop_max_iters)


def build_lpm_raw_path_file(cfg: Dict[str, Any]) -> Path:
    """
    Build non-deduplicated LPM path file.

    Example:
      data/bpic2015/interim/LPM_Path/lpm_group_paths_loop2.json
    """
    model_cfg = cfg["model_based"]

    path_dir = resolve_path(cfg, model_cfg["path_dir"])
    loop_max_iters = int(model_cfg["loop_constraint"]["loop_max_iters"])

    template = model_cfg["path_output"]["raw_template"]

    return path_dir / template.format(loop_max_iters=loop_max_iters)


def build_session_topk_file(
    cfg: Dict[str, Any],
    *,
    k: int,
) -> Path:
    """
    Build session Top-K pattern file.

    Important
    ---------
    session_based.output.dir is already dataset-specific.

    Example:
      data/bpic2015/interim/cut_exports/session_dbscan_cut_topK_K060.json
    """
    session_cfg = cfg["session_based"]

    cut_dir = resolve_path(cfg, session_cfg["output"]["dir"])
    template = session_cfg["output"]["topk_pattern_template"]

    return cut_dir / template.format(K=int(k))