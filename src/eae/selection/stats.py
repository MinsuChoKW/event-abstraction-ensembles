# src/eae/selection/stats.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

import pandas as pd


def build_case_summary_dataframe(
    case_summaries: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert case summaries to DataFrame.
    """
    return pd.DataFrame(case_summaries)


def build_stats_summary_dataframe(
    case_summaries: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build one-row global summary.
    """
    if not case_summaries:
        return pd.DataFrame(
            [
                {
                    "n_cases": 0,
                    "n_reaches_end": 0,
                    "mean_best_coverage_ratio": 0.0,
                    "mean_best_total_cost": None,
                    "mean_best_n_segments": None,
                }
            ]
        )

    df = pd.DataFrame(case_summaries)

    out = {
        "n_cases": int(len(df)),
        "n_reaches_end": int(df["reaches_end"].sum())
        if "reaches_end" in df.columns
        else 0,
        "mean_best_coverage_ratio": float(df["best_coverage_ratio"].mean())
        if "best_coverage_ratio" in df.columns
        else 0.0,
        "mean_best_total_cost": float(df["best_total_cost"].dropna().mean())
        if "best_total_cost" in df.columns and df["best_total_cost"].notna().any()
        else None,
        "mean_best_n_segments": float(df["best_n_segments"].dropna().mean())
        if "best_n_segments" in df.columns and df["best_n_segments"].notna().any()
        else None,
    }

    return pd.DataFrame([out])


def build_label_usage_dataframe(
    selected_rows: Iterable[Dict[str, Any]],
    *,
    top_rank_only: bool = False,
) -> pd.DataFrame:
    """
    Count label usage in selected combinations.

    Parameters
    ----------
    top_rank_only
        If True, only rank-1 combination per case is counted.
        If False, all saved top combinations are counted.
    """
    counter = Counter()

    for row in selected_rows:
        combinations = row.get("combinations", [])

        for comb in combinations:
            if top_rank_only and int(comb.get("rank", -1)) != 1:
                continue

            for label in comb.get("label_sequence", []):
                counter[str(label)] += 1

    rows = [
        {
            "label": label,
            "usage_count": count,
        }
        for label, count in sorted(counter.items())
    ]

    return pd.DataFrame(rows)