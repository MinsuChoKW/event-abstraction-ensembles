from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_jsonl_gz(path: str | Path) -> Iterable[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL.GZ file does not exist: {path}")

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare_original_event_dataframe(
    df: pd.DataFrame,
    *,
    case_id_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Prepare original events for timestamp-based analysis."""
    required = {case_id_column, timestamp_column}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing original-log columns: {sorted(missing)}")

    tmp = df[[case_id_column, timestamp_column]].copy()
    tmp[case_id_column] = tmp[case_id_column].astype(str)
    tmp[timestamp_column] = pd.to_datetime(
        tmp[timestamp_column],
        errors="coerce",
        utc=True,
    )
    tmp = tmp.dropna(subset=[case_id_column, timestamp_column])
    tmp = tmp.sort_values(
        [case_id_column, timestamp_column],
        kind="mergesort",
    ).reset_index(drop=True)
    tmp["event_position"] = (
        tmp.groupby(case_id_column, sort=False).cumcount()
    )
    return tmp


def get_original_inter_event_seconds(
    original_events: pd.DataFrame,
    *,
    case_id_column: str,
    timestamp_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Return non-negative and strictly positive original-event gaps."""
    tmp = original_events[[case_id_column, timestamp_column]].copy()
    tmp["prev_time"] = (
        tmp.groupby(case_id_column, sort=False)[timestamp_column].shift(1)
    )
    tmp["gap_seconds"] = (
        tmp[timestamp_column] - tmp["prev_time"]
    ).dt.total_seconds()

    gaps_raw = tmp["gap_seconds"].dropna()
    gaps_raw = gaps_raw[gaps_raw >= 0]
    return gaps_raw, gaps_raw[gaps_raw > 0]


def load_selected_rank_rows(
    selected_combinations_path: str | Path,
    *,
    rank: int,
) -> Dict[str, Dict[str, Any]]:
    """Load one selected decomposition per case for the requested rank."""
    selected: Dict[str, Dict[str, Any]] = {}

    for row in _read_jsonl_gz(selected_combinations_path):
        if int(row.get("rank", -1)) != int(rank):
            continue
        selected[str(row["case_id"])] = row

    if not selected:
        raise ValueError(
            f"No selected combinations found for rank={rank}: "
            f"{selected_combinations_path}"
        )
    return selected


def build_abstracted_segment_dataframe(
    original_events: pd.DataFrame,
    selected_solution_map: Dict[str, Dict[str, Any]],
    *,
    case_id_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Reconstruct abstracted segment timestamps from selected edges."""
    case_event_times: Dict[str, list[pd.Timestamp]] = {}

    for case_id, group in original_events.groupby(case_id_column, sort=False):
        ordered = group.sort_values("event_position", kind="mergesort")
        case_event_times[str(case_id)] = ordered[timestamp_column].tolist()

    rows: list[Dict[str, Any]] = []
    missing_case_ids: list[str] = []
    invalid_edges: list[str] = []

    for case_id, solution in selected_solution_map.items():
        event_times = case_event_times.get(str(case_id))
        if event_times is None:
            missing_case_ids.append(str(case_id))
            continue

        edges = sorted(
            solution.get("edges", []),
            key=lambda edge: (int(edge["i"]), int(edge["j"])),
        )

        for seg_order, edge in enumerate(edges):
            start_i = int(edge["i"])
            end_j = int(edge["j"])

            if start_i < 0 or end_j <= start_i or end_j > len(event_times):
                invalid_edges.append(
                    f"case={case_id}, edge=[{start_i},{end_j}), "
                    f"case_len={len(event_times)}"
                )
                continue

            rows.append(
                {
                    "case_id": str(case_id),
                    "seg_order": int(seg_order),
                    "seg_start_i": start_i,
                    "seg_end_j": end_j,
                    "seg_start_time": event_times[start_i],
                    "seg_end_time": event_times[end_j - 1],
                    "label": str(edge.get("label", "")),
                }
            )

    if missing_case_ids:
        print(
            "[WARN] Selected cases missing from original log: "
            f"{len(missing_case_ids)}; sample={missing_case_ids[:10]}"
        )

    if invalid_edges:
        raise ValueError(
            "Invalid selected edges were found while reconstructing "
            "segment timestamps.\n"
            + "\n".join(invalid_edges[:10])
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No abstracted segments could be reconstructed.")
    return result


def get_abstracted_inter_event_seconds(
    df_segments: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return non-negative and positive gaps between abstracted segments."""
    required = {
        "case_id",
        "seg_order",
        "seg_start_time",
        "seg_end_time",
    }
    missing = required - set(df_segments.columns)
    if missing:
        raise KeyError(f"Missing segment columns: {sorted(missing)}")

    tmp = df_segments[
        ["case_id", "seg_order", "seg_start_time", "seg_end_time"]
    ].copy()
    tmp = tmp.sort_values(
        ["case_id", "seg_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    tmp["next_seg_start"] = (
        tmp.groupby("case_id", sort=False)["seg_start_time"].shift(-1)
    )
    tmp["gap_seconds"] = (
        tmp["next_seg_start"] - tmp["seg_end_time"]
    ).dt.total_seconds()

    gaps_raw = tmp["gap_seconds"].dropna()
    gaps_raw = gaps_raw[gaps_raw >= 0]
    return gaps_raw, gaps_raw[gaps_raw > 0]


def clip_positive_gaps_at_quantile(
    gaps: pd.Series,
    *,
    quantile: float = 0.99,
) -> tuple[pd.Series, float]:
    """Clip positive gaps at the specified upper quantile."""
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in the interval (0, 1].")
    if gaps.empty:
        return gaps.copy(), float("nan")

    upper = float(gaps.quantile(quantile))
    return gaps[gaps <= upper], upper


def plot_inter_event_time_distribution(
    original_gaps: pd.Series,
    abstracted_gaps: pd.Series,
    *,
    output_dir: str | Path,
    output_stem: str,
    bins_count: int = 70,
    show: bool = False,
) -> Dict[str, Path]:
    """Plot clipped positive inter-event gap distributions."""
    if bins_count < 2:
        raise ValueError("bins_count must be at least 2.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

    for ax, (_, gaps) in zip(
        axes,
        [("Original", original_gaps), ("Abstracted", abstracted_gaps)],
    ):
        if gaps.empty:
            ax.text(
                0.5,
                0.5,
                "No positive gaps",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xlabel("Inter-event time gap", fontsize=15, labelpad=10)
            ax.set_ylabel("Number of event pairs", fontsize=15, labelpad=10)
            ax.tick_params(axis="x", labelsize=13)
            ax.tick_params(axis="y", labelsize=13)
            continue

        min_gap = float(gaps.min())
        max_gap = float(gaps.max())
        if min_gap == max_gap:
            min_gap = max(min_gap * 0.9, 1e-9)
            max_gap = max_gap * 1.1

        bins = np.logspace(
            np.log10(min_gap),
            np.log10(max_gap),
            int(bins_count),
        )

        median_gap = float(gaps.median())
        q25 = float(gaps.quantile(0.25))
        q75 = float(gaps.quantile(0.75))

        ax.hist(
            gaps,
            bins=bins,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.85,
        )
        ax.axvline(
            median_gap,
            linestyle="--",
            linewidth=1.4,
            label=f"Median = {median_gap:.1f}s",
        )
        ax.axvspan(
            q25,
            q75,
            alpha=0.15,
            label=f"IQR = {q25:.1f}s–{q75:.1f}s",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Inter-event time gap", fontsize=15, labelpad=10)
        ax.set_ylabel("Number of event pairs", fontsize=15, labelpad=10)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(bottom=0.28, wspace=0.28)

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}
