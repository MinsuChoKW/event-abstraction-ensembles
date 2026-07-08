from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageDraw

from .bpmn_comparison import build_side_by_side_comparison, save_bpmn_visualizations


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def overlay_abs_bpmn_by_label_prefix(
    base_png_path: str | Path,
    layout_json_path: str | Path,
    meta_json_path: str | Path,
    output_png_path: str | Path,
    *,
    scale_supersample: int = 3,
    node_width: int = 2,
) -> dict[str, Any]:
    """
    Paint activity nodes on an ABS BPMN image using label prefixes:
      - labels starting with 'C' -> red
      - labels starting with 'G' -> blue
    """
    base_png_path = Path(base_png_path)
    layout_json_path = Path(layout_json_path)
    meta_json_path = Path(meta_json_path)
    output_png_path = Path(output_png_path)

    layout = _load_json(layout_json_path)
    meta = _load_json(meta_json_path)

    base_img = Image.open(base_png_path).convert("RGBA")
    img_w, img_h = base_img.size

    graph_w = float(layout["graph"]["width"])
    graph_h = float(layout["graph"]["height"])
    layout_nodes = layout["nodes"]
    meta_nodes = meta["nodes"]

    red_outline = (220, 30, 30, 255)
    red_fill = (255, 120, 120, 90)

    blue_outline = (40, 90, 220, 255)
    blue_fill = (120, 170, 255, 90)

    def gv_to_px(x, y):
        px = (float(x) / graph_w) * (img_w * scale_supersample)
        py = (img_h * scale_supersample) - (
            float(y) / graph_h
        ) * (img_h * scale_supersample)
        return px, py

    def gv_len_x(w):
        return (float(w) / graph_w) * (img_w * scale_supersample)

    def gv_len_y(h):
        return (float(h) / graph_h) * (img_h * scale_supersample)

    big_base = base_img.resize(
        (
            img_w * scale_supersample,
            img_h * scale_supersample,
        ),
        Image.Resampling.LANCZOS,
    )
    overlay = Image.new(
        "RGBA",
        big_base.size,
        (255, 255, 255, 0),
    )
    draw = ImageDraw.Draw(overlay, "RGBA")

    count_c = 0
    count_g = 0
    count_missing_layout = 0

    for node_id, record in meta_nodes.items():
        if record.get("kind") != "activity":
            continue

        label = str(record.get("label", "")).strip()

        for stop in [
            " fontsize=",
            " shape=",
            " style=",
            " fillcolor=",
            " color=",
            " penwidth=",
            " fontname=",
        ]:
            if stop in label:
                label = label.split(stop, 1)[0].strip()

        if len(label) >= 2 and label[0] == '"' and label[-1] == '"':
            label = label[1:-1].strip()

        if label.startswith("C"):
            outline_color = red_outline
            fill_color = red_fill
            count_c += 1
        elif label.startswith("G"):
            outline_color = blue_outline
            fill_color = blue_fill
            count_g += 1
        else:
            continue

        node_layout = layout_nodes.get(node_id)
        if not node_layout:
            count_missing_layout += 1
            continue

        cx, cy = gv_to_px(node_layout["x"], node_layout["y"])
        width = gv_len_x(node_layout["width"])
        height = gv_len_y(node_layout["height"])
        box = [
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        ]

        draw.rounded_rectangle(
            box,
            radius=max(4, int(10 * scale_supersample)),
            fill=fill_color,
            outline=outline_color,
            width=max(1, node_width * scale_supersample),
        )

    merged = Image.alpha_composite(big_base, overlay)
    out = merged.resize((img_w, img_h), Image.Resampling.LANCZOS)

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_png_path)

    return {
        "output_png": output_png_path,
        "count_c": count_c,
        "count_g": count_g,
        "count_missing_layout": count_missing_layout,
    }


def save_abs_bpmn_overlay_visualization(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
) -> dict[str, Path | int]:
    """
    End-to-end helper:
      1) render base BPMN visualizations and assets
      2) create ABS overlay image
      3) create ORG-vs-ABS-overlay comparison image
    """
    base_outputs = save_bpmn_visualizations(
        cfg,
        k=k,
        jump=jump,
        rank=rank,
    )

    overlay_png = (
        base_outputs["output_dir"]
        / "ABS_bpmn_base_C_red_G_blue.png"
    )

    overlay_stats = overlay_abs_bpmn_by_label_prefix(
        base_outputs["abs_png"],
        base_outputs["abs_layout_json"],
        base_outputs["abs_meta_json"],
        overlay_png,
    )

    overlay_comparison_png = build_side_by_side_comparison(
        base_outputs["org_png"],
        overlay_png,
        base_outputs["output_dir"] / f"BPMN_overlay_comparison_rank{rank}.png",
        right_title="ABS BPMN (C red / G blue)",
    )

    return {
        **base_outputs,
        "abs_overlay_png": overlay_stats["output_png"],
        "overlay_comparison_png": overlay_comparison_png,
        "count_c": overlay_stats["count_c"],
        "count_g": overlay_stats["count_g"],
        "count_missing_layout": overlay_stats["count_missing_layout"],
    }
