from __future__ import annotations

import argparse

from eae.config import load_config
from eae.visualization.bpmn_overlay import (
    save_abs_bpmn_overlay_visualization,
)
from eae.visualization.org_pattern_regions import (
    save_org_pattern_region_overlays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BPMN visualizations for one abstraction experiment."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--k",
        required=True,
        type=int,
        help="Target K value.",
    )
    parser.add_argument(
        "--jump",
        required=True,
        type=int,
        help="Target jump allowance.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Target solution/model rank. Default: 1.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the three ORG-region images if possible.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Existing ABS visualization pipeline.
    abs_outputs = save_abs_bpmn_overlay_visualization(
        cfg,
        k=args.k,
        jump=args.jump,
        rank=args.rank,
    )

    # New ORG pattern-region pipeline.
    # Only the three final region PNGs are persisted by this stage.
    region_outputs = save_org_pattern_region_overlays(
        cfg,
        k=args.k,
        jump=args.jump,
        rank=args.rank,
    )

    print("=" * 80)
    print("[VISUALIZATION COMPARISON COMPLETED]")
    print("=" * 80)

    print("[ABS LABEL-PREFIX OVERLAY]")
    print(
        f"ABS overlay PNG        : "
        f"{abs_outputs['abs_overlay_png']}"
    )
    print(
        f"Overlay comparison PNG : "
        f"{abs_outputs['overlay_comparison_png']}"
    )

    print("-" * 80)
    print("[ORG PATTERN REGIONS]")
    print(
        f"G region PNG           : "
        f"{region_outputs['g_region_png']}"
    )
    print(
        f"Neutral region PNG     : "
        f"{region_outputs['neutral_region_png']}"
    )
    print(
        f"C region PNG           : "
        f"{region_outputs['c_region_png']}"
    )

    print("-" * 80)
    print("[MATCH SUMMARY]")
    print(
        f"Unique C segments      : "
        f"{region_outputs['n_unique_c_segments']}"
    )
    print(
        f"Unique G segments      : "
        f"{region_outputs['n_unique_g_segments']}"
    )
    print(
        f"Matched C segments     : "
        f"{region_outputs['matched_c_segments']}"
    )
    print(
        f"Matched G segments     : "
        f"{region_outputs['matched_g_segments']}"
    )
    print(
        f"Unmatched C segments   : "
        f"{region_outputs['unmatched_c_segments']}"
    )
    print(
        f"Unmatched G segments   : "
        f"{region_outputs['unmatched_g_segments']}"
    )

    print("-" * 80)
    print("[REGION OBJECT COUNTS]")
    print(
        f"G nodes / edges        : "
        f"{region_outputs['g_nodes']} / "
        f"{region_outputs['g_edges']}"
    )
    print(
        f"Neutral nodes / edges  : "
        f"{region_outputs['neutral_nodes']} / "
        f"{region_outputs['neutral_edges']}"
    )
    print(
        f"C nodes / edges        : "
        f"{region_outputs['c_nodes']} / "
        f"{region_outputs['c_edges']}"
    )
    print("=" * 80)

    if args.show:
        try:
            from PIL import Image

            for key in (
                "g_region_png",
                "neutral_region_png",
                "c_region_png",
            ):
                Image.open(
                    region_outputs[key]
                ).show()
        except Exception as exc:
            print(
                "[WARN] Could not open region images automatically: "
                f"{exc}"
            )


if __name__ == "__main__":
    main()
