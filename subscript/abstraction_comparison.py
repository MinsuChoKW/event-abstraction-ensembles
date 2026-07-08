from __future__ import annotations

import argparse

from eae.config import load_config
from eae.information import (
    build_abstracted_segment_dataframe,
    build_bpmn_size_summary,
    build_original_abstraction_summary,
    clip_positive_gaps_at_quantile,
    compute_case_lengths,
    format_summary_table,
    get_abstracted_inter_event_seconds,
    get_original_inter_event_seconds,
    load_selected_rank_rows,
    load_xes_dataframe,
    plot_case_length_distribution,
    plot_inter_event_time_distribution,
    prepare_original_event_dataframe,
    resolve_bpmn_paths,
    resolve_experiment_inputs,
    save_bpmn_size_summary,
    save_summary_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate EDA figures and information tables for one "
            "event-abstraction experiment."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--jump", required=True, type=int)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--case-length-bins", type=int, default=50)
    parser.add_argument("--inter-event-bins", type=int, default=70)
    parser.add_argument("--inter-event-quantile", type=float, default=0.99)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    inputs = resolve_experiment_inputs(
        cfg,
        k=args.k,
        jump=args.jump,
        rank=args.rank,
    )

    dataset_cfg = cfg["dataset"]
    case_id_column = dataset_cfg["columns"]["case_id"]
    activity_column = dataset_cfg["columns"]["activity"]
    timestamp_column = dataset_cfg["columns"]["timestamp"]

    original_df = load_xes_dataframe(inputs["original_log"])
    abstracted_df = load_xes_dataframe(inputs["abstracted_log"])

    output_dir = inputs["run_dir"] / "figures" / "eda"

    # 1) BPMN size summary
    original_bpmn_path, abstracted_bpmn_path = resolve_bpmn_paths(
        run_dir=inputs["run_dir"],
        rank=args.rank,
    )
    bpmn_numeric_table, bpmn_display_table = build_bpmn_size_summary(
        original_bpmn_path,
        abstracted_bpmn_path,
    )
    bpmn_summary_paths = save_bpmn_size_summary(
        bpmn_numeric_table,
        bpmn_display_table,
        output_dir=output_dir,
        output_stem=(
            f"bpmn_size_summary_K{args.k:03d}_"
            f"jump{args.jump:02d}_rank{args.rank}"
        ),
    )

    # 2) Case-length distribution
    original_lengths = compute_case_lengths(
        original_df,
        case_id_column=case_id_column,
    )
    abstracted_lengths = compute_case_lengths(
        abstracted_df,
        case_id_column=case_id_column,
    )
    case_length_paths = plot_case_length_distribution(
        original_lengths,
        abstracted_lengths,
        output_dir=output_dir,
        output_stem=(
            f"case_length_distribution_K{args.k:03d}_"
            f"jump{args.jump:02d}_rank{args.rank}_log"
        ),
        bins_count=args.case_length_bins,
        show=args.show,
    )

    # Shared segment reconstruction for outputs 3 and 4
    prepared_original_df = prepare_original_event_dataframe(
        original_df,
        case_id_column=case_id_column,
        timestamp_column=timestamp_column,
    )
    selected_solution_map = load_selected_rank_rows(
        inputs["selected_combinations"],
        rank=args.rank,
    )
    segment_df = build_abstracted_segment_dataframe(
        prepared_original_df,
        selected_solution_map,
        case_id_column=case_id_column,
        timestamp_column=timestamp_column,
    )

    # 3) Original-vs-abstracted log summary
    summary_table = build_original_abstraction_summary(
        original_df,
        segment_df,
        case_id_column=case_id_column,
        activity_column=activity_column,
        timestamp_column=timestamp_column,
    )
    summary_display = format_summary_table(summary_table)
    summary_paths = save_summary_table(
        summary_table,
        output_dir=output_dir,
        output_stem=(
            f"original_vs_abstracted_summary_K{args.k:03d}_"
            f"jump{args.jump:02d}_rank{args.rank}"
        ),
    )

    # 4) Inter-event time distribution
    original_gaps_raw, original_gaps_positive = (
        get_original_inter_event_seconds(
            prepared_original_df,
            case_id_column=case_id_column,
            timestamp_column=timestamp_column,
        )
    )
    abstracted_gaps_raw, abstracted_gaps_positive = (
        get_abstracted_inter_event_seconds(segment_df)
    )

    original_gaps_clipped, original_upper = (
        clip_positive_gaps_at_quantile(
            original_gaps_positive,
            quantile=args.inter_event_quantile,
        )
    )
    abstracted_gaps_clipped, abstracted_upper = (
        clip_positive_gaps_at_quantile(
            abstracted_gaps_positive,
            quantile=args.inter_event_quantile,
        )
    )

    inter_event_paths = plot_inter_event_time_distribution(
        original_gaps_clipped,
        abstracted_gaps_clipped,
        output_dir=output_dir,
        output_stem=(
            f"inter_event_time_distribution_K{args.k:03d}_"
            f"jump{args.jump:02d}_rank{args.rank}_"
            f"p{int(args.inter_event_quantile * 100):02d}_log"
        ),
        bins_count=args.inter_event_bins,
        show=args.show,
    )

    original_zero_ratio = (
        float((original_gaps_raw == 0).mean() * 100)
        if len(original_gaps_raw)
        else float("nan")
    )
    abstracted_zero_ratio = (
        float((abstracted_gaps_raw == 0).mean() * 100)
        if len(abstracted_gaps_raw)
        else float("nan")
    )

    print("=" * 80)
    print("[EDA] Completed")
    print("=" * 80)
    print(
        f"[RUN] dataset={cfg['dataset']['name']}, "
        f"K={args.k}, jump={args.jump}, rank={args.rank}"
    )

    print("-" * 80)
    print("[BPMN SIZE SUMMARY]")
    print(bpmn_display_table.to_string())
    print(f"Numeric CSV   : {bpmn_summary_paths['numeric_csv']}")
    print(f"Formatted CSV : {bpmn_summary_paths['formatted_csv']}")

    print("-" * 80)
    print("[ORIGINAL VS ABSTRACTED SUMMARY]")
    print(summary_display.to_string())
    print(f"Numeric CSV   : {summary_paths['numeric_csv']}")
    print(f"Formatted CSV : {summary_paths['formatted_csv']}")

    print("-" * 80)
    print("[CASE LENGTH]")
    print(f"Original mean   : {original_lengths.mean():.4f}")
    print(f"Abstracted mean : {abstracted_lengths.mean():.4f}")
    print(f"PNG             : {case_length_paths['png']}")
    print(f"PDF             : {case_length_paths['pdf']}")

    print("-" * 80)
    print("[INTER-EVENT TIME]")
    print(f"Original zero ratio   : {original_zero_ratio:.4f}%")
    print(f"Abstracted zero ratio : {abstracted_zero_ratio:.4f}%")
    print(f"Original upper q      : {original_upper:.4f}s")
    print(f"Abstracted upper q    : {abstracted_upper:.4f}s")
    print(f"PNG                    : {inter_event_paths['png']}")
    print(f"PDF                    : {inter_event_paths['pdf']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
