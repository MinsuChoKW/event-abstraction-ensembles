#/project_root/src/eae/information/sweep_graph.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from eae.paths import build_run_dir, resolve_path


STYLE_ENSEMBLE_HIGHLIGHT = {
    "Ensemble": {
        "label": "Ensemble",
        "color": "#d62728",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 3.3,
        "markersize": 6.0,
        "alpha": 1.00,
        "zorder": 5,
    },
    "Session": {
        "label": "Session",
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 4.6,
        "alpha": 0.78,
        "zorder": 3,
    },
}


def safe_filename(value: str) -> str:
    """Convert an arbitrary label to a filesystem-safe name."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _single_int_value(
    values: Any,
    *,
    field_name: str,
) -> int:
    """
    Read one fixed integer from a YAML list.

    A fixed-k sweep config must contain exactly one K value.
    """
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(
            f"{field_name} must be a list containing exactly one value. "
            f"Received: {values!r}"
        )
    return int(values[0])


def _integer_list(
    values: Any,
    *,
    field_name: str,
) -> list[int]:
    """Read, validate, deduplicate, and sort an integer list."""
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"{field_name} must be a non-empty list. "
            f"Received: {values!r}"
        )

    parsed = sorted({int(value) for value in values})
    if any(value < 0 for value in parsed):
        raise ValueError(
            f"{field_name} cannot contain negative values: {parsed}"
        )
    return parsed


def validate_sweep_configs(
    ensemble_cfg: Dict[str, Any],
    session_cfg: Dict[str, Any],
) -> tuple[str, int, list[int]]:
    """
    Validate that two configs describe comparable fixed-K jump sweeps.

    Returns
    -------
    dataset_name
    fixed_k
    jump_values
    """
    ensemble_dataset = str(ensemble_cfg["dataset"]["name"])
    session_dataset = str(session_cfg["dataset"]["name"])

    if ensemble_dataset != session_dataset:
        raise ValueError(
            "The two configs must use the same dataset. "
            f"Ensemble={ensemble_dataset}, Session={session_dataset}"
        )

    ensemble_method = str(
        ensemble_cfg["abstraction_source"]["method"]
    ).upper()
    session_method = str(
        session_cfg["abstraction_source"]["method"]
    ).upper()

    if ensemble_method != "BOTH":
        raise ValueError(
            "The ensemble config must use "
            "abstraction_source.method: BOTH. "
            f"Received: {ensemble_method}"
        )

    if session_method != "SESSION":
        raise ValueError(
            "The session config must use "
            "abstraction_source.method: SESSION. "
            f"Received: {session_method}"
        )

    ensemble_k = _single_int_value(
        ensemble_cfg["abstraction"]["k_values"],
        field_name="ensemble abstraction.k_values",
    )
    session_k = _single_int_value(
        session_cfg["abstraction"]["k_values"],
        field_name="session abstraction.k_values",
    )

    if ensemble_k != session_k:
        raise ValueError(
            "The two configs must use the same fixed K. "
            f"Ensemble={ensemble_k}, Session={session_k}"
        )

    ensemble_jumps = _integer_list(
        ensemble_cfg["abstraction"]["jump_values"],
        field_name="ensemble abstraction.jump_values",
    )
    session_jumps = _integer_list(
        session_cfg["abstraction"]["jump_values"],
        field_name="session abstraction.jump_values",
    )

    if ensemble_jumps != session_jumps:
        raise ValueError(
            "The two configs must contain the same jump sweep values. "
            f"Ensemble={ensemble_jumps}, Session={session_jumps}"
        )

    return ensemble_dataset, ensemble_k, ensemble_jumps


def _read_f1_from_conformance_summary(
    csv_path: str | Path,
    *,
    rank: int,
    model_type: str = "EXP",
) -> dict[str, float]:
    """
    Read one conformance row from a Stage 06 summary.

    EXP is used by default because the jump-sweep figure evaluates the
    expanded model against the corresponding original-prefix log.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            "Conformance summary does not exist. "
            "Run Stage 06 first:\n"
            f"{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "model_type",
        "precision",
        "recall_fitness",
        "f1",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns in {csv_path}: {sorted(missing)}"
        )

    selected = df[
        df["model_type"].astype(str).str.upper()
        == str(model_type).upper()
    ].copy()

    if "rank" in selected.columns:
        numeric_rank = pd.to_numeric(
            selected["rank"],
            errors="coerce",
        )
        ranked = selected[numeric_rank == int(rank)]
        if not ranked.empty:
            selected = ranked

    if selected.empty:
        raise ValueError(
            f"No model_type={model_type}, rank={rank} row found in "
            f"{csv_path}"
        )

    # Defensive duplicate handling: retain the row with the highest F1.
    selected["f1"] = pd.to_numeric(
        selected["f1"],
        errors="coerce",
    )
    selected = selected.dropna(subset=["f1"])

    if selected.empty:
        raise ValueError(
            f"No numeric F1 value found in {csv_path}"
        )

    row = selected.sort_values("f1").iloc[-1]

    return {
        "fitness": float(row["recall_fitness"]),
        "precision": float(row["precision"]),
        "f1": float(row["f1"]),
    }


def collect_fixed_k_jump_sweep(
    cfg: Dict[str, Any],
    *,
    strategy: str,
    rank: int = 1,
    model_type: str = "EXP",
) -> pd.DataFrame:
    """
    Collect Stage 06 conformance results for every configured jump value.
    """
    fixed_k = _single_int_value(
        cfg["abstraction"]["k_values"],
        field_name="abstraction.k_values",
    )
    jump_values = _integer_list(
        cfg["abstraction"]["jump_values"],
        field_name="abstraction.jump_values",
    )

    rows: list[dict[str, Any]] = []

    for jump in jump_values:
        run_dir = build_run_dir(
            cfg,
            jump=jump,
            k=fixed_k,
        )
        summary_path = (
            run_dir
            / "evaluation"
            / f"conformance_summary_rank{rank}.csv"
        )

        metrics = _read_f1_from_conformance_summary(
            summary_path,
            rank=rank,
            model_type=model_type,
        )

        rows.append(
            {
                "strategy": strategy,
                "fixed_k": fixed_k,
                "jump": jump,
                "fitness": metrics["fitness"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "source_csv": str(summary_path),
            }
        )

    return pd.DataFrame(rows)


def collect_two_config_sweep(
    ensemble_cfg: Dict[str, Any],
    session_cfg: Dict[str, Any],
    *,
    rank: int = 1,
    model_type: str = "EXP",
) -> tuple[pd.DataFrame, str, int]:
    """
    Collect and combine Ensemble and Session fixed-K jump sweeps.
    """
    dataset_name, fixed_k, _ = validate_sweep_configs(
        ensemble_cfg,
        session_cfg,
    )

    ensemble_df = collect_fixed_k_jump_sweep(
        ensemble_cfg,
        strategy="Ensemble",
        rank=rank,
        model_type=model_type,
    )
    session_df = collect_fixed_k_jump_sweep(
        session_cfg,
        strategy="Session",
        rank=rank,
        model_type=model_type,
    )

    combined = pd.concat(
        [ensemble_df, session_df],
        ignore_index=True,
    )

    combined = (
        combined.sort_values(
            ["strategy", "jump", "f1"]
        )
        .drop_duplicates(
            subset=["strategy", "jump"],
            keep="last",
        )
        .sort_values(["strategy", "jump"])
        .reset_index(drop=True)
    )

    return combined, dataset_name, fixed_k


def default_sweep_output_dir(
    cfg: Dict[str, Any],
    *,
    dataset_name: str,
) -> Path:
    """Return a shared output directory outside individual run folders."""
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
        / "sweeps"
    )


def plot_jump_sweep_f1(
    sweep_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    dataset_name: str,
    fixed_k: int,
    show: bool = False,
    style_dict: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Path]:
    """
    Plot the red Ensemble / blue Session F1 jump-sweep figure.
    """
    required = {"strategy", "jump", "f1"}
    missing = required - set(sweep_df.columns)
    if missing:
        raise KeyError(
            f"Missing sweep columns: {sorted(missing)}"
        )

    style_dict = (
        STYLE_ENSEMBLE_HIGHLIGHT
        if style_dict is None
        else style_dict
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    for strategy in ("Ensemble", "Session"):
        sub = sweep_df[
            sweep_df["strategy"] == strategy
        ].copy()
        sub["f1"] = pd.to_numeric(
            sub["f1"],
            errors="coerce",
        )
        sub = sub.dropna(subset=["f1"]).sort_values("jump")

        if sub.empty:
            raise ValueError(
                f"No valid F1 rows found for strategy={strategy}"
            )

        style = style_dict[strategy]

        ax.plot(
            sub["jump"].to_numpy(),
            sub["f1"].to_numpy(),
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            color=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=style["label"],
        )

    ensemble_rows = sweep_df[
        sweep_df["strategy"] == "Ensemble"
    ].copy()
    ensemble_rows["f1"] = pd.to_numeric(
        ensemble_rows["f1"],
        errors="coerce",
    )
    ensemble_rows = ensemble_rows.dropna(subset=["f1"])

    if not ensemble_rows.empty:
        best_index = ensemble_rows["f1"].idxmax()
        best_jump = int(
            ensemble_rows.loc[best_index, "jump"]
        )
        best_f1 = float(
            ensemble_rows.loc[best_index, "f1"]
        )

        ax.scatter(
            [best_jump],
            [best_f1],
            s=100,
            color=style_dict["Ensemble"]["color"],
            edgecolor="black",
            linewidth=0.8,
            zorder=10,
        )

    ax.set_xlabel("Jump allowance", fontsize=18)
    ax.set_ylabel("F1-Score", fontsize=18)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="both", linestyle=":", alpha=0.55)
    ax.legend(loc="best", fontsize=13, frameon=True)

    jump_values = sorted(
        sweep_df["jump"].dropna().astype(int).unique()
    )
    ax.set_xticks(jump_values)
    fig.tight_layout()

    dataset_slug = safe_filename(dataset_name)
    output_stem = (
        f"figure_{dataset_slug}_fixed_k{fixed_k:03d}_"
        "jump_allowance_sweep_f1_ensemble_highlight"
    )

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    csv_path = output_dir / (
        f"{dataset_slug}_fixed_k{fixed_k:03d}_"
        "jump_sweep_metrics.csv"
    )

    sweep_df.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return {
        "csv": csv_path,
        "png": png_path,
        "pdf": pdf_path,
    }


def best_ensemble_row(sweep_df: pd.DataFrame) -> pd.Series:
    """Return the highest-F1 Ensemble row."""
    ensemble = sweep_df[
        sweep_df["strategy"] == "Ensemble"
    ].copy()
    ensemble["f1"] = pd.to_numeric(
        ensemble["f1"],
        errors="coerce",
    )
    ensemble = ensemble.dropna(subset=["f1"])

    if ensemble.empty:
        raise ValueError("No Ensemble F1 results are available.")

    return ensemble.loc[ensemble["f1"].idxmax()]
