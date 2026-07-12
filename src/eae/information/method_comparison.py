#/project_root/src/eae/information/method_comparison.py
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eae.paths import build_selection_output_paths, resolve_path


STRATEGY_ORDER = ["LPM", "Session", "Ensemble"]


def _single_int_value(
    values: Any,
    *,
    field_name: str,
) -> int:
    """Read one fixed integer value from a YAML list."""
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(
            f"{field_name} must contain exactly one value. "
            f"Received: {values!r}"
        )
    return int(values[0])


def _open_text_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _edge_cost_sum(edges: list[dict[str, Any]]) -> float:
    costs: list[float] = []

    for edge in edges:
        for key in ("cost", "c", "align_cost"):
            value = edge.get(key)
            if value is not None:
                costs.append(float(value))
                break

    return float(np.sum(costs)) if costs else float("nan")


def _read_selected_rows(
    selected_path: str | Path,
    *,
    strategy: str,
    dataset_name: str,
    fixed_k: int,
    jump: int,
    rank: int,
) -> pd.DataFrame:
    """
    Read one selected rank from a Stage 04 selected-combinations JSONL file.

    Stage 04 stores one row per selected solution:
      case_id, rank, case_len, total_cost, edges, ...

    Therefore --rank 1 means:
      use only the first-ranked decomposition per case,
      exactly as Stage 05 discovery does.
    """
    selected_path = Path(selected_path)

    if not selected_path.exists():
        raise FileNotFoundError(
            "Selected-combinations file does not exist. "
            "Run Stage 04 first:\n"
            f"{selected_path}"
        )

    rows: list[dict[str, Any]] = []

    with _open_text_auto(selected_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            row_rank = obj.get("rank")
            if row_rank is not None and int(row_rank) != int(rank):
                continue

            case_id = obj.get(
                "case_id",
                obj.get("case", obj.get("trace_id", line_number)),
            )

            # IMPORTANT:
            # Stage 04 solution_to_dict() writes "case_len", not "case_length".
            case_length = obj.get(
                "n",
                obj.get(
                    "case_len",
                    obj.get("case_length", obj.get("length")),
                ),
            )

            solved = bool(obj.get("solved", True))
            edges = obj.get("edges") or []

            if obj.get("total_cost") is not None:
                align_cost = float(obj["total_cost"])
            elif obj.get("align_cost") is not None:
                align_cost = float(obj["align_cost"])
            elif obj.get("dsum") is not None:
                align_cost = float(obj["dsum"])
            elif obj.get("cost") is not None:
                align_cost = float(obj["cost"])
            else:
                align_cost = _edge_cost_sum(edges)

            labels_used = []
            for edge in edges:
                label = edge.get(
                    "label",
                    edge.get("l", edge.get("pattern")),
                )
                if label is not None:
                    labels_used.append(str(label))

            rows.append(
                {
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "case_id": str(case_id),
                    "case_length": case_length,
                    "align_cost": align_cost,
                    "n_segments": int(
                        obj.get("n_segments", len(labels_used))
                    ),
                    "terminal_pos": obj.get(
                        "terminal_pos",
                        obj.get("end_i"),
                    ),
                    "coverage": obj.get("coverage"),
                    "coverage_ratio": obj.get("coverage_ratio"),
                    "delta": obj.get("delta"),
                    "solved": solved,
                    "fixed_k": int(fixed_k),
                    "jump": int(jump),
                    "rank": int(rank),
                    "source_file": str(selected_path),
                }
            )

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise ValueError(
            f"No rank={rank} rows were found in {selected_path}"
        )

    frame["case_length"] = pd.to_numeric(
        frame["case_length"],
        errors="coerce",
    )
    frame["align_cost"] = pd.to_numeric(
        frame["align_cost"],
        errors="coerce",
    )

    for col in ["terminal_pos", "coverage", "coverage_ratio", "delta"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(
                frame[col],
                errors="coerce",
            )

    before_filter = len(frame)

    frame = frame[
        frame["solved"]
        & frame["case_length"].notna()
        & frame["align_cost"].notna()
    ].copy()

    if frame.empty:
        raise ValueError(
            f"No valid solved rows were found in {selected_path}. "
            f"rows_before_filter={before_filter}, "
            f"case_length_notna="
            f"{pd.to_numeric(pd.DataFrame(rows)['case_length'], errors='coerce').notna().sum()}, "
            f"align_cost_notna="
            f"{pd.to_numeric(pd.DataFrame(rows)['align_cost'], errors='coerce').notna().sum()}"
        )

    frame["case_length"] = frame["case_length"].astype(int)
    frame["align_cost"] = frame["align_cost"].astype(float)

    frame = (
        frame.sort_values(["case_id", "align_cost"])
        .drop_duplicates(subset=["case_id"], keep="first")
        .reset_index(drop=True)
    )

    return frame


def _validate_method_configs(
    ensemble_cfg: Dict[str, Any],
    session_cfg: Dict[str, Any],
    lpm_cfg: Dict[str, Any],
) -> str:
    """Validate dataset and abstraction-source roles."""
    configs = {
        "Ensemble": ensemble_cfg,
        "Session": session_cfg,
        "LPM": lpm_cfg,
    }

    dataset_names = {
        str(cfg["dataset"]["name"])
        for cfg in configs.values()
    }
    if len(dataset_names) != 1:
        raise ValueError(
            "All three configs must use the same dataset. "
            f"Received: {sorted(dataset_names)}"
        )

    expected_methods = {
        "Ensemble": "BOTH",
        "Session": "SESSION",
        "LPM": "LPM",
    }

    for strategy, cfg in configs.items():
        actual = str(
            cfg["abstraction_source"]["method"]
        ).upper()
        expected = expected_methods[strategy]

        if actual != expected:
            raise ValueError(
                f"{strategy} config must use "
                f"abstraction_source.method: {expected}. "
                f"Received: {actual}"
            )

    return next(iter(dataset_names))


def collect_three_method_results(
    ensemble_cfg: Dict[str, Any],
    session_cfg: Dict[str, Any],
    lpm_cfg: Dict[str, Any],
    *,
    ensemble_jump: int,
    session_jump: int,
    lpm_jump: int,
    rank: int = 1,
) -> tuple[pd.DataFrame, str]:
    """
    Collect per-case alignment results for Ensemble, Session, and LPM.
    """
    dataset_name = _validate_method_configs(
        ensemble_cfg,
        session_cfg,
        lpm_cfg,
    )

    specs = [
        (
            "Ensemble",
            ensemble_cfg,
            int(ensemble_jump),
        ),
        (
            "Session",
            session_cfg,
            int(session_jump),
        ),
        (
            "LPM",
            lpm_cfg,
            int(lpm_jump),
        ),
    ]

    parts: list[pd.DataFrame] = []

    for strategy, cfg, jump in specs:
        fixed_k = _single_int_value(
            cfg["abstraction"]["k_values"],
            field_name=f"{strategy} abstraction.k_values",
        )

        selection_paths = build_selection_output_paths(
            cfg,
            jump=jump,
            k=fixed_k,
        )

        selected_path = selection_paths["selected_combinations"]

        part = _read_selected_rows(
            selected_path,
            strategy=strategy,
            dataset_name=dataset_name,
            fixed_k=fixed_k,
            jump=jump,
            rank=rank,
        )
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)

    for strategy in STRATEGY_ORDER:
        if combined[combined["strategy"] == strategy].empty:
            raise ValueError(
                f"No valid rows were collected for {strategy}."
            )

    return combined, dataset_name


def default_method_comparison_output_dir(
    cfg: Dict[str, Any],
    *,
    dataset_name: str,
) -> Path:
    """Return the shared method-comparison output directory."""
    results_root = resolve_path(
        cfg,
        cfg["results"]["root_dir"],
    )
    figures_dir = str(
        cfg["results"].get("figures_dir", "figures")
    )

    return (
        results_root
        / figures_dir
        / dataset_name
        / "method_comparison"
    )


def build_alignment_distribution_table(
    per_case_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-case alignment-cost distribution table."""
    rows: list[dict[str, Any]] = []

    for strategy in STRATEGY_ORDER:
        values = (
            per_case_df.loc[
                per_case_df["strategy"] == strategy,
                "align_cost",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(values) == 0:
            continue

        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))

        rows.append(
            {
                "strategy": strategy,
                "n_cases": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }
        )

    return pd.DataFrame(rows)


def plot_alignment_distribution(
    per_case_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    dataset_name: str,
    show: bool = False,
) -> dict[str, Path]:
    """Create the LPM/Session/Ensemble alignment-cost boxplot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    labels = []

    for strategy in STRATEGY_ORDER:
        values = (
            per_case_df.loc[
                per_case_df["strategy"] == strategy,
                "align_cost",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values):
            data.append(values)
            labels.append(strategy)

    if len(data) != 3:
        raise ValueError(
            "All three strategies are required for the boxplot."
        )

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    boxplot = ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        widths=0.55,
    )

    for patch in boxplot["boxes"]:
        patch.set_facecolor("#dde6f0")
        patch.set_edgecolor("#3a4a5b")

    for median in boxplot["medians"]:
        median.set_color("#1f3a5f")

    for mean in boxplot["means"]:
        mean.set_color("#a23b1d")

    ax.set_ylabel(
        "Per-case alignment cost",
        fontsize=15,
        labelpad=10,
    )
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()

    stem = (
        f"figure_{dataset_name.lower()}_"
        "per_case_alignment_distribution_by_method"
    )
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}


def _moving_average_smoother(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    if len(x_sorted) < n_bins:
        return x_sorted, y_sorted

    splits = np.array_split(
        np.arange(len(x_sorted)),
        n_bins,
    )

    x_centers = np.array(
        [x_sorted[idx].mean() for idx in splits if len(idx)]
    )
    y_centers = np.array(
        [y_sorted[idx].mean() for idx in splits if len(idx)]
    )

    return x_centers, y_centers


def assign_length_groups(
    per_case_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """
    Assign short/medium/long groups using Ensemble case-length tertiles.
    """
    reference = per_case_df[
        per_case_df["strategy"] == "Ensemble"
    ].copy()

    if reference.empty:
        raise ValueError(
            "Ensemble rows are required to define length tertiles."
        )

    boundary_1, boundary_2 = np.quantile(
        reference["case_length"].to_numpy(dtype=float),
        [1 / 3, 2 / 3],
    )

    def assign(length: float) -> str:
        if length <= boundary_1:
            return "short"
        if length <= boundary_2:
            return "medium"
        return "long"

    result = per_case_df.copy()
    result["length_group"] = result["case_length"].map(assign)

    return result, float(boundary_1), float(boundary_2)


def build_length_stratified_table(
    per_case_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Build the short/medium/long alignment-cost table."""
    stratified, boundary_1, boundary_2 = assign_length_groups(
        per_case_df
    )

    rows: list[dict[str, Any]] = []

    for group in ("short", "medium", "long"):
        for strategy in STRATEGY_ORDER:
            subset = stratified[
                (stratified["length_group"] == group)
                & (stratified["strategy"] == strategy)
            ]

            if subset.empty:
                continue

            rows.append(
                {
                    "strategy": strategy,
                    "length_group": group,
                    "length_low": int(
                        subset["case_length"].min()
                    ),
                    "length_high": int(
                        subset["case_length"].max()
                    ),
                    "n_cases": int(len(subset)),
                    "mean_align_cost": float(
                        subset["align_cost"].mean()
                    ),
                    "median_align_cost": float(
                        subset["align_cost"].median()
                    ),
                    "p95_align_cost": float(
                        subset["align_cost"].quantile(0.95)
                    ),
                }
            )

    return (
        pd.DataFrame(rows),
        boundary_1,
        boundary_2,
    )


def plot_case_length_alignment_cost(
    per_case_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    dataset_name: str,
    show: bool = False,
    n_bins: int = 20,
) -> dict[str, Path]:
    """Plot alignment cost against case length for all three methods."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style = {
        "LPM": {
            "marker_size": 6,
            "alpha": 0.14,
            "linewidth": 2.0,
        },
        "Session": {
            "marker_size": 6,
            "alpha": 0.14,
            "linewidth": 2.0,
        },
        "Ensemble": {
            "marker_size": 6,
            "alpha": 0.14,
            "linewidth": 2.2,
        },
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for strategy in STRATEGY_ORDER:
        subset = per_case_df[
            per_case_df["strategy"] == strategy
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No rows available for strategy={strategy}"
            )

        x = subset["case_length"].to_numpy(dtype=float)
        y = subset["align_cost"].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            s=style[strategy]["marker_size"],
            alpha=style[strategy]["alpha"],
            rasterized=True,
        )

        x_center, y_center = _moving_average_smoother(
            x,
            y,
            n_bins=n_bins,
        )

        ax.plot(
            x_center,
            y_center,
            linewidth=style[strategy]["linewidth"],
            label=strategy,
        )

    ax.set_xlabel(
        r"Case length $|\sigma|$",
        fontsize=15,
        labelpad=10,
    )
    ax.set_ylabel(
        "Per-case alignment cost",
        fontsize=15,
        labelpad=10,
    )
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(
        fontsize=13,
        markerscale=1.8,
        handlelength=2.4,
        borderpad=0.8,
        labelspacing=0.6,
    )

    fig.tight_layout()

    stem = (
        f"figure_{dataset_name.lower()}_"
        "case_length_stratified_alignment_cost_by_method"
    )
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}


def save_method_comparison_outputs(
    per_case_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    dataset_name: str,
    show: bool = False,
) -> dict[str, Any]:
    """
    Save the two requested tables and figures.

    No label-utilization figures are produced.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution_table = build_alignment_distribution_table(
        per_case_df
    )
    distribution_csv = (
        output_dir
        / "table_per_case_alignment_distribution_by_method.csv"
    )
    distribution_table.to_csv(distribution_csv, index=False)

    stratified_table, boundary_1, boundary_2 = (
        build_length_stratified_table(per_case_df)
    )
    stratified_csv = (
        output_dir
        / "table_case_length_stratified_alignment_cost_by_method.csv"
    )
    stratified_table.to_csv(stratified_csv, index=False)

    distribution_figure = plot_alignment_distribution(
        per_case_df,
        output_dir=output_dir,
        dataset_name=dataset_name,
        show=show,
    )
    stratified_figure = plot_case_length_alignment_cost(
        per_case_df,
        output_dir=output_dir,
        dataset_name=dataset_name,
        show=show,
    )

    return {
        "distribution_table": distribution_table,
        "distribution_csv": distribution_csv,
        "distribution_figure": distribution_figure,
        "stratified_table": stratified_table,
        "stratified_csv": stratified_csv,
        "stratified_figure": stratified_figure,
        "tertile_boundary_1": boundary_1,
        "tertile_boundary_2": boundary_2,
    }
