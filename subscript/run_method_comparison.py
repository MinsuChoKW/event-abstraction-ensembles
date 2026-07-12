#/project_root/subscript/run_method_comparison.py
from __future__ import annotations

import argparse
from pathlib import Path

from eae.config import load_config
from eae.information.method_comparison import (
    collect_three_method_results,
    default_method_comparison_output_dir,
    save_method_comparison_outputs,
)
from eae.information.sweep_graph import (
    best_ensemble_row,
    collect_two_config_sweep,
    default_sweep_output_dir,
    plot_jump_sweep_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Ensemble-vs-Session jump sweep and the "
            "three-method alignment-cost comparison."
        )
    )

    parser.add_argument(
        "--ensemble-config",
        required=True,
        help="Config using abstraction_source.method: BOTH.",
    )
    parser.add_argument(
        "--session-config",
        required=True,
        help="Config using abstraction_source.method: SESSION.",
    )
    parser.add_argument(
        "--lpm-config",
        required=True,
        help="Config using abstraction_source.method: LPM.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Selected solution rank. Default: 1.",
    )
    parser.add_argument(
        "--model-type",
        default="EXP",
        choices=["ORG", "ABS", "EXP"],
        help=(
            "Conformance-summary row used for the jump sweep. "
            "Default: EXP."
        ),
    )

    parser.add_argument(
        "--ensemble-compare-jump",
        required=True,
        type=int,
        help=(
            "Ensemble jump used in the three-method comparison."
        ),
    )
    parser.add_argument(
        "--session-compare-jump",
        required=True,
        type=int,
        help=(
            "Session jump used in the three-method comparison."
        ),
    )
    parser.add_argument(
        "--lpm-compare-jump",
        required=True,
        type=int,
        help=(
            "LPM jump used in the three-method comparison."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional common output root. When omitted, outputs are "
            "stored under results/{figures_dir}/{dataset}."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensemble_cfg = load_config(args.ensemble_config)
    session_cfg = load_config(args.session_config)
    lpm_cfg = load_config(args.lpm_config)

    # ========================================================
    # 1. Ensemble vs Session fixed-K jump sweep
    # ========================================================
    sweep_df, sweep_dataset, fixed_k = collect_two_config_sweep(
        ensemble_cfg,
        session_cfg,
        rank=args.rank,
        model_type=args.model_type,
    )

    if args.output_dir:
        common_root = Path(args.output_dir)
        sweep_output_dir = common_root / "sweeps"
    else:
        sweep_output_dir = default_sweep_output_dir(
            ensemble_cfg,
            dataset_name=sweep_dataset,
        )

    sweep_paths = plot_jump_sweep_f1(
        sweep_df,
        output_dir=sweep_output_dir,
        dataset_name=sweep_dataset,
        fixed_k=fixed_k,
        show=args.show,
    )
    best = best_ensemble_row(sweep_df)

    # ========================================================
    # 2. LPM vs Session vs Ensemble method comparison
    # ========================================================
    per_case_df, comparison_dataset = (
        collect_three_method_results(
            ensemble_cfg,
            session_cfg,
            lpm_cfg,
            ensemble_jump=args.ensemble_compare_jump,
            session_jump=args.session_compare_jump,
            lpm_jump=args.lpm_compare_jump,
            rank=args.rank,
        )
    )

    if args.output_dir:
        comparison_output_dir = (
            common_root / "method_comparison"
        )
    else:
        comparison_output_dir = (
            default_method_comparison_output_dir(
                ensemble_cfg,
                dataset_name=comparison_dataset,
            )
        )

    comparison_outputs = save_method_comparison_outputs(
        per_case_df,
        output_dir=comparison_output_dir,
        dataset_name=comparison_dataset,
        show=args.show,
    )

    # ========================================================
    # Console output
    # ========================================================
    print("=" * 80)
    print("[FIXED-K JUMP SWEEP: ENSEMBLE VS SESSION]")
    print("=" * 80)
    print(f"Dataset         : {sweep_dataset}")
    print(f"Fixed K         : {fixed_k}")
    print(f"Rank            : {args.rank}")
    print(f"Model type      : {args.model_type}")
    print()
    print(
        sweep_df[
            [
                "strategy",
                "fixed_k",
                "jump",
                "fitness",
                "precision",
                "f1",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Best Ensemble   : "
        f"jump={int(best['jump'])}, "
        f"F1={float(best['f1']):.6f}"
    )
    print(f"Sweep CSV       : {sweep_paths['csv']}")
    print(f"Sweep PNG       : {sweep_paths['png']}")
    print(f"Sweep PDF       : {sweep_paths['pdf']}")

    print("=" * 80)
    print("[PER-CASE ALIGNMENT DISTRIBUTION]")
    print("=" * 80)
    print(
        comparison_outputs[
            "distribution_table"
        ].to_string(index=False)
    )
    print(
        f"Table CSV       : "
        f"{comparison_outputs['distribution_csv']}"
    )
    print(
        f"Figure PNG      : "
        f"{comparison_outputs['distribution_figure']['png']}"
    )
    print(
        f"Figure PDF      : "
        f"{comparison_outputs['distribution_figure']['pdf']}"
    )

    print("=" * 80)
    print("[CASE-LENGTH-STRATIFIED ALIGNMENT COST]")
    print("=" * 80)
    print(
        "Tertile bounds  : "
        f"{comparison_outputs['tertile_boundary_1']:.4f}, "
        f"{comparison_outputs['tertile_boundary_2']:.4f}"
    )
    print(
        comparison_outputs[
            "stratified_table"
        ].to_string(index=False)
    )
    print(
        f"Table CSV       : "
        f"{comparison_outputs['stratified_csv']}"
    )
    print(
        f"Figure PNG      : "
        f"{comparison_outputs['stratified_figure']['png']}"
    )
    print(
        f"Figure PDF      : "
        f"{comparison_outputs['stratified_figure']['pdf']}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
