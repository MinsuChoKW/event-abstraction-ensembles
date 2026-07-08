from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pm4py

from eae.paths import (
    build_run_dir,
    build_selection_output_paths,
    resolve_path,
)


def load_xes_dataframe(xes_path: str | Path) -> pd.DataFrame:
    """Load an XES event log as a pandas DataFrame."""
    xes_path = Path(xes_path)
    if not xes_path.exists():
        raise FileNotFoundError(f"XES file does not exist: {xes_path}")

    log = pm4py.read_xes(str(xes_path))
    if isinstance(log, pd.DataFrame):
        return log.copy()
    return pm4py.convert_to_dataframe(log)


def resolve_experiment_inputs(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int,
) -> Dict[str, Path]:
    """Resolve the original log, abstracted log, and selection output."""
    original_log_path = resolve_path(cfg, cfg["dataset"]["raw_log_path"])
    run_dir = build_run_dir(cfg, jump=jump, k=k)
    abstracted_log_path = (
        run_dir / "discovery" / f"abstracted_log_rank{rank}.xes"
    )

    selection_paths = build_selection_output_paths(
        cfg,
        jump=jump,
        k=k,
    )
    selected_combinations_path = selection_paths["selected_combinations"]

    required = {
        "original_log": original_log_path,
        "abstracted_log": abstracted_log_path,
        "selected_combinations": selected_combinations_path,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required experiment inputs are missing:\n"
            + "\n".join(missing)
            + "\n"
            + f"Requested experiment: K={k}, jump={jump}, rank={rank}"
        )

    return {**required, "run_dir": run_dir}


def build_original_abstraction_summary(
    original_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    *,
    case_id_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Build the compact original-vs-abstracted log summary table."""
    original_required = {
        case_id_column,
        activity_column,
        timestamp_column,
    }
    missing_original = original_required - set(original_df.columns)
    if missing_original:
        raise KeyError(
            f"Missing columns in original log: {sorted(missing_original)}"
        )

    segment_required = {"case_id", "seg_order", "label"}
    missing_segments = segment_required - set(segment_df.columns)
    if missing_segments:
        raise KeyError(
            f"Missing columns in segment DataFrame: {sorted(missing_segments)}"
        )

    original_sorted = original_df.copy()
    original_sorted[case_id_column] = (
        original_sorted[case_id_column].astype(str)
    )
    original_sorted[timestamp_column] = pd.to_datetime(
        original_sorted[timestamp_column],
        errors="coerce",
        utc=True,
    )
    original_sorted = original_sorted.sort_values(
        [case_id_column, timestamp_column],
        kind="mergesort",
    ).reset_index(drop=True)

    original_case_lengths = (
        original_sorted.groupby(case_id_column, sort=False).size()
    )
    original_variants = (
        original_sorted.groupby(case_id_column, sort=False)[activity_column]
        .apply(lambda values: tuple(values.astype(str).tolist()))
    )

    abstracted_sorted = segment_df.copy()
    abstracted_sorted["case_id"] = abstracted_sorted["case_id"].astype(str)
    abstracted_sorted = abstracted_sorted.sort_values(
        ["case_id", "seg_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    abstracted_case_lengths = (
        abstracted_sorted.groupby("case_id", sort=False).size()
    )
    abstracted_variants = (
        abstracted_sorted.groupby("case_id", sort=False)["label"]
        .apply(lambda values: tuple(values.astype(str).tolist()))
    )

    summary = pd.DataFrame(
        {
            "Original": {
                "Cases": int(original_sorted[case_id_column].nunique()),
                "Events": int(len(original_sorted)),
                "Activities": int(
                    original_sorted[activity_column].nunique()
                ),
                "Variants": int(original_variants.nunique()),
                "Avg. case length": float(original_case_lengths.mean()),
                "Min. case length": int(original_case_lengths.min()),
                "Max. case length": int(original_case_lengths.max()),
            },
            "Abstracted": {
                "Cases": int(abstracted_sorted["case_id"].nunique()),
                "Events": int(len(abstracted_sorted)),
                "Activities": int(abstracted_sorted["label"].nunique()),
                "Variants": int(abstracted_variants.nunique()),
                "Avg. case length": float(abstracted_case_lengths.mean()),
                "Min. case length": int(abstracted_case_lengths.min()),
                "Max. case length": int(abstracted_case_lengths.max()),
            },
        }
    )

    return summary.loc[
        [
            "Cases",
            "Events",
            "Activities",
            "Variants",
            "Avg. case length",
            "Min. case length",
            "Max. case length",
        ]
    ]


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Format a summary table for display and manuscript inspection."""
    formatted = summary.copy().astype(object)

    for metric in [
        "Cases",
        "Events",
        "Activities",
        "Variants",
        "Min. case length",
        "Max. case length",
    ]:
        formatted.loc[metric] = summary.loc[metric].map(
            lambda value: f"{int(round(value)):,}"
        )

    formatted.loc["Avg. case length"] = (
        summary.loc["Avg. case length"]
        .map(lambda value: f"{float(value):.2f}")
    )
    return formatted


def save_summary_table(
    summary: pd.DataFrame,
    *,
    output_dir: str | Path,
    output_stem: str,
) -> dict[str, Path]:
    """Save numeric and formatted summary tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_csv_path = output_dir / f"{output_stem}.csv"
    formatted_csv_path = output_dir / f"{output_stem}_formatted.csv"

    summary.to_csv(
        numeric_csv_path,
        index=True,
        index_label="Metric",
    )
    format_summary_table(summary).to_csv(
        formatted_csv_path,
        index=True,
        index_label="Metric",
    )
    return {
        "numeric_csv": numeric_csv_path,
        "formatted_csv": formatted_csv_path,
    }
