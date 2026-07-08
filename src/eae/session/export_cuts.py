# src/eae/session/export_cuts.py

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def save_json_gz(
    obj: Any,
    path: str | Path,
) -> Path:
    """
    Save an object as gzipped JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

    return path


def load_json_gz(
    path: str | Path,
) -> Any:
    """
    Load gzipped JSON.
    """
    path = Path(path)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    obj: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """
    Save an object as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

    return path


def build_dataset_cut_dir(
    cfg: Dict[str, Any],
) -> Path:
    """
    Build session cut output directory.

    Important
    ---------
    session_based.output.dir is already dataset-specific.

    Config:
      session_based.output.dir: data/bpic2015/interim/cut_exports

    Output:
      data/bpic2015/interim/cut_exports
    """
    from eae.config import resolve_path

    return resolve_path(cfg, cfg["session_based"]["output"]["dir"])


def build_session_output_paths(
    cfg: Dict[str, Any],
) -> Dict[str, Path]:
    """
    Build session output paths.
    """
    out_cfg = cfg["session_based"]["output"]
    cut_dir = build_dataset_cut_dir(cfg)
    cut_dir.mkdir(parents=True, exist_ok=True)

    return {
        "cut_dir": cut_dir,
        "session_info": cut_dir / out_cfg["session_info_file"],
        "cluster_info": cut_dir / out_cfg["cluster_info_file"],
        "cluster_patterns": cut_dir / out_cfg["cluster_patterns_file"],
        "report": cut_dir / "session_pipeline_report.json",
    }


def build_topk_pattern_payload(
    cluster_patterns: List[Dict[str, Any]],
    *,
    k: int,
    dataset_name: str,
    label_prefix: str = "S",
    include_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build K-dependent session pattern file.

    Output schema
    -------------
    {
      "meta": {...},
      "patterns": [
        {
          "label": "S001",
          "rank": 1,
          "cluster_label": 0,
          "support": 10,
          "sequence": [...]
        }
      ]
    }
    """
    if k <= 0:
        raise ValueError(f"K must be positive, got {k}")

    selected = cluster_patterns[: int(k)]
    patterns = []

    for i, row in enumerate(selected, start=1):
        item = dict(row)
        item["label"] = f"{label_prefix}{i:03d}"
        item["rank"] = int(i)
        patterns.append(item)

    meta = {
        "dataset": dataset_name,
        "created_at": datetime.now().isoformat(),
        "k": int(k),
        "label_prefix": label_prefix,
        "n_patterns": len(patterns),
    }

    if include_meta:
        meta.update(include_meta)

    return {
        "meta": meta,
        "patterns": patterns,
    }


def export_topk_session_pattern_files(
    cfg: Dict[str, Any],
    cluster_patterns: List[Dict[str, Any]],
) -> List[Path]:
    """
    Export one Top-K pattern JSON file for each K in config.

    Files are saved under:
      session_based.output.dir

    Example:
      data/bpic2015/interim/cut_exports/
    """
    dataset_name = cfg["dataset"]["name"]
    cut_dir = build_dataset_cut_dir(cfg)
    cut_dir.mkdir(parents=True, exist_ok=True)

    k_values = cfg["abstraction"]["k_values"]
    template = cfg["session_based"]["output"]["topk_pattern_template"]
    label_prefix = cfg["abstraction_source"]["label_prefix"]["session"]

    clustering_cfg = cfg["session_based"]["clustering"]
    session_cfg = cfg["session_based"]["session_split"]

    written = []

    for k in k_values:
        k = int(k)

        payload = build_topk_pattern_payload(
            cluster_patterns,
            k=k,
            dataset_name=dataset_name,
            label_prefix=label_prefix,
            include_meta={
                "gap_hours": session_cfg["gap_hours"],
                "clustering_algorithm": clustering_cfg["algorithm"],
                "eps": clustering_cfg["eps"],
                "min_samples": clustering_cfg["min_samples"],
                "metric": clustering_cfg["metric"],
            },
        )

        filename = template.format(K=k)
        path = cut_dir / filename
        save_json(payload, path)
        written.append(path)

    return written