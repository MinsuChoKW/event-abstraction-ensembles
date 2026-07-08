from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict

import pm4py
from PIL import Image, ImageOps, ImageDraw

from eae.paths import build_run_dir


def ensure_graphviz_available() -> None:
    """Ensure that Graphviz is available."""
    try:
        subprocess.check_output(
            ["dot", "-V"],
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        raise RuntimeError(
            "Graphviz ('dot') is not available. "
            "Please install graphviz first."
        ) from exc


def _read_bpmn_graph(bpmn_path: str | Path):
    """Read a BPMN file with pm4py."""
    bpmn_path = Path(bpmn_path)

    if not bpmn_path.exists():
        raise FileNotFoundError(
            f"BPMN file does not exist: {bpmn_path}"
        )

    try:
        return pm4py.read_bpmn(str(bpmn_path))
    except Exception:
        from pm4py.objects.bpmn.importer import importer as bpmn_importer
        return bpmn_importer.apply(str(bpmn_path))


def resolve_bpmn_model_paths(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
) -> Dict[str, Path]:
    """
    Resolve ORG and ABS BPMN paths for one experiment.
    """
    run_dir = build_run_dir(
        cfg,
        jump=jump,
        k=k,
    )

    candidates_org = [
        run_dir / "evaluation" / "complexity" / "ORG_model.bpmn",
        run_dir / "complexity" / "ORG_model.bpmn",
    ]
    candidates_abs = [
        run_dir / "evaluation" / "complexity" / f"ABS_model_rank{rank}.bpmn",
        run_dir / "complexity" / f"ABS_model_rank{rank}.bpmn",
    ]

    export_dir = run_dir / "bpmn_exports"
    if export_dir.exists():
        candidates_org.extend(sorted(export_dir.glob("*ORG*.bpmn")))
        candidates_abs.extend(sorted(export_dir.glob("*ABS*.bpmn")))

    org_path = next((p for p in candidates_org if p.exists()), None)
    abs_path = next((p for p in candidates_abs if p.exists()), None)

    if org_path is None or abs_path is None:
        missing = []
        if org_path is None:
            missing.append("ORG BPMN not found")
        if abs_path is None:
            missing.append("ABS BPMN not found")

        raise FileNotFoundError(
            "Could not resolve BPMN model paths for visualization.\n"
            f"Run directory: {run_dir}\n"
            + "\n".join(missing)
        )

    return {
        "run_dir": run_dir,
        "org_bpmn": org_path,
        "abs_bpmn": abs_path,
    }


def _parse_plain_layout(plain_text: str) -> dict[str, Any]:
    """
    Parse Graphviz 'dot -Tplain' output into a minimal JSON layout.
    """
    layout: dict[str, Any] = {
        "graph": {},
        "nodes": {},
        "edges": [],
    }

    for line in plain_text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        kind = parts[0]

        if kind == "graph" and len(parts) >= 4:
            layout["graph"] = {
                "scale": float(parts[1]),
                "width": float(parts[2]),
                "height": float(parts[3]),
            }

        elif kind == "node" and len(parts) >= 6:
            node_id = parts[1]
            layout["nodes"][node_id] = {
                "x": float(parts[2]),
                "y": float(parts[3]),
                "width": float(parts[4]),
                "height": float(parts[5]),
            }

        elif kind == "edge" and len(parts) >= 4:
            tail = parts[1]
            head = parts[2]
            n_points = int(parts[3])

            coords = []
            for i in range(n_points):
                x = float(parts[4 + 2 * i])
                y = float(parts[5 + 2 * i])
                coords.append([x, y])

            layout["edges"].append(
                {
                    "tail": tail,
                    "head": head,
                    "points": coords,
                }
            )

    if not layout["graph"]:
        raise ValueError(
            "Failed to parse Graphviz plain layout."
        )

    return layout


def _parse_dot_source_to_meta(dot_source: str) -> dict[str, Any]:
    """
    Parse node and edge metadata from Graphviz DOT source.

    This is intentionally lightweight and captures only the fields
    needed for image overlays.
    """
    meta_nodes: dict[str, Any] = {}
    meta_edges: list[dict[str, Any]] = []

    node_pattern = re.compile(
        r'^\s*("?[^"\s]+"?)\s+\[(.+)\]\s*$'
    )
    edge_pattern = re.compile(
        r'^\s*("?[^"\s]+"?)\s*->\s*("?[^"\s]+"?)'
    )

    def strip_quotes(text: str) -> str:
        text = text.strip()
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            return text[1:-1]
        return text

    def extract_attr(attr_text: str, key: str):
        match = re.search(
            rf'{key}=(".*?"|[^,\]]+)',
            attr_text,
        )
        if not match:
            return None
        value = match.group(1).strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        return value

    seen_edges: set[tuple[str, str]] = set()

    for raw_line in dot_source.splitlines():
        line = raw_line.strip().rstrip(";")

        if not line or line in {"digraph {", "{", "}"}:
            continue

        if "->" in line:
            match = edge_pattern.search(line)
            if match:
                source = strip_quotes(match.group(1))
                target = strip_quotes(match.group(2))
                if (source, target) not in seen_edges:
                    seen_edges.add((source, target))
                    meta_edges.append(
                        {
                            "id": f"{source}__{target}",
                            "source": source,
                            "target": target,
                            "orig_source": source,
                            "orig_target": target,
                        }
                    )
            continue

        match = node_pattern.search(line)
        if not match:
            continue

        node_id = strip_quotes(match.group(1))
        attr_text = match.group(2)

        label = extract_attr(attr_text, "label") or ""
        shape = (extract_attr(attr_text, "shape") or "").lower()

        if shape == "box":
            kind = "activity"
        elif shape == "diamond":
            kind = "gateway"
        elif shape == "doublecircle":
            kind = "end_event"
        elif shape == "circle":
            kind = "event" if label else "start_event"
        else:
            kind = "other"

        meta_nodes[node_id] = {
            "id": node_id,
            "orig_id": node_id,
            "label": label,
            "class_name": shape,
            "kind": kind,
        }

    return {
        "nodes": meta_nodes,
        "edges": meta_edges,
    }


def _save_dot_plain_and_meta(
    dot_source: str,
    *,
    dot_path: str | Path,
    layout_json_path: str | Path,
    meta_json_path: str | Path,
) -> None:
    """
    Save DOT source, Graphviz plain layout, and parsed metadata.
    """
    dot_path = Path(dot_path)
    layout_json_path = Path(layout_json_path)
    meta_json_path = Path(meta_json_path)

    dot_path.parent.mkdir(parents=True, exist_ok=True)

    dot_path.write_text(dot_source, encoding="utf-8")

    result = subprocess.run(
        ["dot", "-Tplain", str(dot_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    layout = _parse_plain_layout(result.stdout)
    meta = _parse_dot_source_to_meta(dot_source)

    layout_json_path.write_text(
        json.dumps(layout, indent=2),
        encoding="utf-8",
    )
    meta_json_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def render_bpmn_png(
    bpmn_path: str | Path,
    output_png: str | Path,
) -> Path:
    """
    Render one BPMN file to PNG using pm4py.
    """
    ensure_graphviz_available()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    bpmn_graph = _read_bpmn_graph(bpmn_path)
    from pm4py.visualization.bpmn import visualizer as bpmn_vis

    gviz = bpmn_vis.apply(bpmn_graph)

    try:
        bpmn_vis.save(gviz, str(output_png))
    except Exception:
        gviz.render(
            filename=str(output_png.with_suffix("")),
            format="png",
            cleanup=True,
        )

    if not output_png.exists():
        raise RuntimeError(
            f"Failed to render BPMN PNG: {output_png}"
        )

    return output_png


def render_bpmn_with_assets(
    bpmn_path: str | Path,
    *,
    output_png: str | Path,
    output_dot: str | Path,
    output_layout_json: str | Path,
    output_meta_json: str | Path,
) -> dict[str, Path]:
    """
    Render one BPMN file to PNG and save overlay-support assets.
    """
    ensure_graphviz_available()

    output_png = Path(output_png)
    output_dot = Path(output_dot)
    output_layout_json = Path(output_layout_json)
    output_meta_json = Path(output_meta_json)

    output_png.parent.mkdir(parents=True, exist_ok=True)

    bpmn_graph = _read_bpmn_graph(bpmn_path)
    from pm4py.visualization.bpmn import visualizer as bpmn_vis

    gviz = bpmn_vis.apply(bpmn_graph)

    try:
        bpmn_vis.save(gviz, str(output_png))
    except Exception:
        gviz.render(
            filename=str(output_png.with_suffix("")),
            format="png",
            cleanup=True,
        )

    if not output_png.exists():
        raise RuntimeError(
            f"Failed to render BPMN PNG: {output_png}"
        )

    _save_dot_plain_and_meta(
        gviz.source,
        dot_path=output_dot,
        layout_json_path=output_layout_json,
        meta_json_path=output_meta_json,
    )

    return {
        "png": output_png,
        "dot": output_dot,
        "layout_json": output_layout_json,
        "meta_json": output_meta_json,
    }


def build_side_by_side_comparison(
    org_png_path: str | Path,
    abs_png_path: str | Path,
    output_png: str | Path,
    *,
    margin: int = 24,
    header_height: int = 48,
    right_title: str = "ABS BPMN",
) -> Path:
    """
    Create a side-by-side comparison image from ORG and ABS PNGs.
    """
    org_png_path = Path(org_png_path)
    abs_png_path = Path(abs_png_path)
    output_png = Path(output_png)

    if not org_png_path.exists():
        raise FileNotFoundError(f"Missing PNG: {org_png_path}")
    if not abs_png_path.exists():
        raise FileNotFoundError(f"Missing PNG: {abs_png_path}")

    org_img = Image.open(org_png_path).convert("RGB")
    abs_img = Image.open(abs_png_path).convert("RGB")

    org_img = ImageOps.expand(org_img, border=0, fill="white")
    abs_img = ImageOps.expand(abs_img, border=0, fill="white")

    target_height = max(org_img.height, abs_img.height)

    def pad_to_height(image: Image.Image, height: int) -> Image.Image:
        if image.height == height:
            return image
        top = (height - image.height) // 2
        bottom = height - image.height - top
        return ImageOps.expand(
            image,
            border=(0, top, 0, bottom),
            fill="white",
        )

    org_img = pad_to_height(org_img, target_height)
    abs_img = pad_to_height(abs_img, target_height)

    total_width = org_img.width + abs_img.width + margin * 3
    total_height = target_height + margin * 2 + header_height

    canvas = Image.new(
        "RGB",
        (total_width, total_height),
        color="white",
    )

    org_xy = (margin, margin + header_height)
    abs_xy = (org_img.width + margin * 2, margin + header_height)
    canvas.paste(org_img, org_xy)
    canvas.paste(abs_img, abs_xy)

    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "ORG BPMN", fill="black")
    draw.text((org_img.width + margin * 2, margin), right_title, fill="black")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)

    return output_png


def save_bpmn_visualizations(
    cfg: Dict[str, Any],
    *,
    k: int,
    jump: int,
    rank: int = 1,
) -> Dict[str, Path]:
    """
    Render ORG and ABS BPMN models and save:
      - base PNGs
      - DOT/layout/meta assets
      - plain ORG-vs-ABS comparison image
    """
    resolved = resolve_bpmn_model_paths(
        cfg,
        k=k,
        jump=jump,
        rank=rank,
    )

    run_dir = resolved["run_dir"]
    output_dir = run_dir / "figures" / "visualization"

    org_assets = render_bpmn_with_assets(
        resolved["org_bpmn"],
        output_png=output_dir / "ORG_bpmn_base.png",
        output_dot=output_dir / "ORG_bpmn_base.dot",
        output_layout_json=output_dir / "ORG_bpmn_base_layout.json",
        output_meta_json=output_dir / "ORG_bpmn_base_meta.json",
    )

    abs_assets = render_bpmn_with_assets(
        resolved["abs_bpmn"],
        output_png=output_dir / "ABS_bpmn_base.png",
        output_dot=output_dir / "ABS_bpmn_base.dot",
        output_layout_json=output_dir / "ABS_bpmn_base_layout.json",
        output_meta_json=output_dir / "ABS_bpmn_base_meta.json",
    )

    comparison_png = build_side_by_side_comparison(
        org_assets["png"],
        abs_assets["png"],
        output_dir / f"BPMN_comparison_rank{rank}.png",
    )

    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "org_bpmn": resolved["org_bpmn"],
        "abs_bpmn": resolved["abs_bpmn"],
        "org_png": org_assets["png"],
        "org_dot": org_assets["dot"],
        "org_layout_json": org_assets["layout_json"],
        "org_meta_json": org_assets["meta_json"],
        "abs_png": abs_assets["png"],
        "abs_dot": abs_assets["dot"],
        "abs_layout_json": abs_assets["layout_json"],
        "abs_meta_json": abs_assets["meta_json"],
        "comparison_png": comparison_png,
    }
