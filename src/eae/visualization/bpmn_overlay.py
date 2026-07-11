from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image, ImageDraw

from .bpmn_comparison import save_bpmn_visualizations


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_prefixes(
    *values: Any,
) -> tuple[str, ...]:
    prefixes: list[str] = []

    for value in values:
        if value is None:
            continue

        if isinstance(value, (list, tuple, set)):
            candidates = value
        else:
            candidates = [value]

        for candidate in candidates:
            text = str(candidate).strip()

            if not text:
                continue

            if text not in prefixes:
                prefixes.append(text)

    return tuple(prefixes)


def _get_family_prefixes(
    cfg: Dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Return prefixes for visual buckets.

    Red bucket:
        instance/session-based labels.
        Supports current config prefix, plus legacy C/S.

    Blue bucket:
        model/LPM-based labels.
        Supports current config prefix, plus legacy G.
    """
    label_prefix = (
        cfg.get("abstraction_source", {})
        .get("label_prefix", {})
    )

    session_prefix = label_prefix.get("session")
    lpm_prefix = label_prefix.get("lpm")

    red_prefixes = _normalize_prefixes(
        "S",
    )
    blue_prefixes = _normalize_prefixes(
        lpm_prefix,
        "G",
    )

    return red_prefixes, blue_prefixes


def _clean_label(raw: Any) -> str:
    label = str(raw).strip()

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

    if (
        len(label) >= 2
        and label[0] == '"'
        and label[-1] == '"'
    ):
        label = label[1:-1].strip()

    return label


def _starts_with_any(
    text: str,
    prefixes: Iterable[str],
) -> bool:
    return any(
        text.startswith(prefix)
        for prefix in prefixes
        if prefix
    )


def overlay_abs_bpmn_by_label_prefix(
    base_png_path: str | Path,
    layout_json_path: str | Path,
    meta_json_path: str | Path,
    output_png_path: str | Path,
    *,
    red_prefixes: tuple[str, ...] = ("S",),
    blue_prefixes: tuple[str, ...] = ("G",),
    scale_supersample: int = 3,
    node_width: int = 2,
) -> dict[str, Any]:
    """
    Paint activity nodes on an ABS BPMN image using label prefixes.

    Red:
        instance/session-side labels, for example C* or S*.

    Blue:
        model/LPM-side labels, for example G*.
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
        px = (float(x) / graph_w) * (
            img_w * scale_supersample
        )
        py = (
            img_h * scale_supersample
        ) - (float(y) / graph_h) * (
            img_h * scale_supersample
        )
        return px, py

    def gv_len_x(width):
        return (float(width) / graph_w) * (
            img_w * scale_supersample
        )

    def gv_len_y(height):
        return (float(height) / graph_h) * (
            img_h * scale_supersample
        )

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

    count_red = 0
    count_blue = 0
    count_missing_layout = 0

    for node_id, record in meta_nodes.items():
        if record.get("kind") != "activity":
            continue

        label = _clean_label(record.get("label", ""))

        if _starts_with_any(label, red_prefixes):
            outline_color = red_outline
            fill_color = red_fill
            count_red += 1
        elif _starts_with_any(label, blue_prefixes):
            outline_color = blue_outline
            fill_color = blue_fill
            count_blue += 1
        else:
            continue

        node_layout = layout_nodes.get(node_id)

        if not node_layout:
            count_missing_layout += 1
            continue

        cx, cy = gv_to_px(
            node_layout["x"],
            node_layout["y"],
        )
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
            radius=max(
                4,
                int(10 * scale_supersample),
            ),
            fill=fill_color,
            outline=outline_color,
            width=max(
                1,
                node_width * scale_supersample,
            ),
        )

    merged = Image.alpha_composite(
        big_base,
        overlay,
    )
    out = merged.resize(
        (img_w, img_h),
        Image.Resampling.LANCZOS,
    )

    output_png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.save(output_png_path)

    base_img.close()
    big_base.close()
    overlay.close()
    merged.close()
    out.close()

    return {
        "output_png": output_png_path,
        "count_red": count_red,
        "count_blue": count_blue,
        "count_missing_layout": count_missing_layout,
        "red_prefixes": red_prefixes,
        "blue_prefixes": blue_prefixes,
    }


def save_abs_bpmn_overlay_visualization(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
) -> dict[str, Path | int | tuple[str, ...]]:
    """
    End-to-end helper:
      1) render ORG and ABS base BPMN visualizations
      2) create ABS prefix overlay image

    No side-by-side comparison image is created here.
    """
    base_outputs = save_bpmn_visualizations(
        cfg,
        k=k,
        jump=jump,
        rank=rank,
    )

    red_prefixes, blue_prefixes = _get_family_prefixes(cfg)

    overlay_png = (
        base_outputs["output_dir"]
        / "ABS_bpmn_abstraction_type_overlay.png"
    )

    overlay_stats = overlay_abs_bpmn_by_label_prefix(
        base_outputs["abs_png"],
        base_outputs["abs_layout_json"],
        base_outputs["abs_meta_json"],
        overlay_png,
        red_prefixes=red_prefixes,
        blue_prefixes=blue_prefixes,
    )

    return {
        **base_outputs,
        "abs_overlay_png": overlay_stats["output_png"],
        "count_red": overlay_stats["count_red"],
        "count_blue": overlay_stats["count_blue"],
        "count_missing_layout": overlay_stats[
            "count_missing_layout"
        ],
        "red_prefixes": overlay_stats["red_prefixes"],
        "blue_prefixes": overlay_stats["blue_prefixes"],
    }