from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_case_lengths(
    event_df: pd.DataFrame,
    *,
    case_id_column: str,
) -> pd.Series:
    """Compute the number of events per case."""
    if case_id_column not in event_df.columns:
        raise KeyError(
            f"Missing case ID column: {case_id_column}\n"
            f"Available columns: {list(event_df.columns)}"
        )

    case_lengths = (
        event_df.groupby(case_id_column, sort=False)
        .size()
        .rename("case_length")
        .astype(float)
    )
    return case_lengths[case_lengths > 0]


def plot_case_length_distribution(
    original_lengths: pd.Series,
    abstracted_lengths: pd.Series,
    *,
    output_dir: str | Path,
    output_stem: str,
    bins_count: int = 50,
    show: bool = False,
) -> Dict[str, Path]:
    """Plot original and abstracted case-length distributions."""
    if original_lengths.empty:
        raise ValueError("The original log has no positive case lengths.")
    if abstracted_lengths.empty:
        raise ValueError("The abstracted log has no positive case lengths.")
    if bins_count < 2:
        raise ValueError("bins_count must be at least 2.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_mean = float(original_lengths.mean())
    abstracted_mean = float(abstracted_lengths.mean())

    x_min = float(min(original_lengths.min(), abstracted_lengths.min()))
    x_max = float(max(original_lengths.max(), abstracted_lengths.max()))

    if x_min == x_max:
        x_min = max(x_min * 0.9, 1e-6)
        x_max = x_max * 1.1

    bins = np.logspace(np.log10(x_min), np.log10(x_max), int(bins_count))

    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.hist(
        original_lengths,
        bins=bins,
        alpha=0.45,
        label=f"Original (mean = {original_mean:.2f})",
        color="tab:blue",
        edgecolor="none",
        zorder=1,
    )
    ax.hist(
        abstracted_lengths,
        bins=bins,
        alpha=0.75,
        label=f"Abstracted (mean = {abstracted_mean:.2f})",
        color="purple",
        edgecolor="none",
        zorder=2,
    )
    ax.axvline(
        original_mean,
        linestyle="--",
        linewidth=1.4,
        color="deepskyblue",
        label=f"Original mean = {original_mean:.2f}",
        zorder=3,
    )
    ax.axvline(
        abstracted_mean,
        linestyle="--",
        linewidth=1.4,
        color="indigo",
        label=f"Abstracted mean = {abstracted_mean:.2f}",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Case length", fontsize=15, labelpad=10)
    ax.set_ylabel("Number of cases", fontsize=15, labelpad=10)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=10)
    fig.subplots_adjust(bottom=0.22)

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}
