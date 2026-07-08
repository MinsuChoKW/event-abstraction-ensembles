# scripts/02_build_session_patterns.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict

from eae.config import load_config, resolve_path

from eae.session.split import build_sessions_from_log
from eae.session.encode import attach_session_sequences, encode_sessions_bow
from eae.session.cluster import (
    attach_cluster_labels,
    build_cluster_pattern_table,
    cluster_sessions_dbscan,
    summarize_clusters,
)
from eae.session.export_cuts import (
    build_session_output_paths,
    export_topk_session_pattern_files,
    save_json,
    save_json_gz,
)


def run_session_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run session-based preprocessing pipeline.

    Steps
    -----
    1. Load raw event log.
    2. Split each case into sessions using gap_hours.
    3. Convert sessions into bag-of-activities vectors.
    4. Cluster sessions with DBSCAN.
    5. Save session-level information.
    6. Save cluster-level information.
    7. Save ranked cluster pattern table.
    8. Generate K-dependent session pattern files using cfg["abstraction"]["k_values"].
    """
    session_cfg = cfg["session_based"]

    if not bool(session_cfg.get("enabled", True)):
        print("[Session] session_based.enabled is false. Skipping session pipeline.")
        return {
            "status": "skipped",
            "reason": "session_based.enabled is false",
        }

    dataset_cfg = cfg["dataset"]
    columns_cfg = dataset_cfg["columns"]

    raw_log_path = resolve_path(cfg, dataset_cfg["raw_log_path"])

    gap_hours = float(session_cfg["session_split"]["gap_hours"])

    clustering_cfg = session_cfg["clustering"]
    algorithm = clustering_cfg["algorithm"].lower()
    eps = float(clustering_cfg["eps"])
    min_samples = int(clustering_cfg["min_samples"])
    metric = clustering_cfg.get("metric", "euclidean")

    if algorithm != "dbscan":
        raise ValueError(
            f"Unsupported session clustering algorithm: {algorithm}. "
            "Currently only dbscan is implemented."
        )

    output_paths = build_session_output_paths(cfg)

    print("=" * 80)
    print("[Session] Start session-based preprocessing")
    print("=" * 80)
    print(f"[Session] dataset        : {dataset_cfg['name']}")
    print(f"[Session] raw_log_path   : {raw_log_path}")
    print(f"[Session] cut_dir        : {output_paths['cut_dir']}")
    print(f"[Session] gap_hours     : {gap_hours}")
    print(f"[Session] algorithm     : {algorithm}")
    print(f"[Session] eps           : {eps}")
    print(f"[Session] min_samples   : {min_samples}")
    print(f"[Session] metric        : {metric}")
    print(f"[Session] k_values      : {cfg['abstraction']['k_values']}")

    # --------------------------------------------------------
    # 1. Build sessions
    # --------------------------------------------------------
    sessions = build_sessions_from_log(
        raw_log_path,
        case_col=columns_cfg["case_id"],
        activity_col=columns_cfg["activity"],
        timestamp_col=columns_cfg["timestamp"],
        gap_hours=gap_hours,
    )

    sessions = attach_session_sequences(sessions)

    print("[Session] n_sessions:", len(sessions))

    # --------------------------------------------------------
    # 2. Encode sessions
    # --------------------------------------------------------
    X, vocab = encode_sessions_bow(
        sessions,
        normalize="l1",
    )

    print("[Session] vocab size:", len(vocab))
    print("[Session] feature shape:", X.shape)

    # --------------------------------------------------------
    # 3. DBSCAN clustering
    # --------------------------------------------------------
    labels = cluster_sessions_dbscan(
        X,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )

    clustered_sessions = attach_cluster_labels(sessions, labels)
    cluster_summary = summarize_clusters(clustered_sessions)

    print("[Session] n_clusters:", cluster_summary["n_clusters"])
    print("[Session] n_noise:", cluster_summary["n_noise"])

    # --------------------------------------------------------
    # 4. Build ranked cluster pattern table
    # --------------------------------------------------------
    cluster_patterns = build_cluster_pattern_table(
        clustered_sessions,
        include_noise=False,
    )

    print("[Session] n_cluster_patterns:", len(cluster_patterns))

    # --------------------------------------------------------
    # 5. Save clustering-level outputs
    # --------------------------------------------------------
    session_info_payload = {
        "meta": {
            "dataset": dataset_cfg["name"],
            "created_at": datetime.now().isoformat(),
            "gap_hours": gap_hours,
            "n_sessions": len(clustered_sessions),
        },
        "sessions": clustered_sessions,
    }

    cluster_info_payload = {
        "meta": {
            "dataset": dataset_cfg["name"],
            "created_at": datetime.now().isoformat(),
            "algorithm": algorithm,
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
        },
        "summary": cluster_summary,
        "activity_vocabulary": vocab,
    }

    cluster_patterns_payload = {
        "meta": {
            "dataset": dataset_cfg["name"],
            "created_at": datetime.now().isoformat(),
            "algorithm": algorithm,
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "n_cluster_patterns": len(cluster_patterns),
        },
        "patterns": cluster_patterns,
    }

    save_json_gz(session_info_payload, output_paths["session_info"])
    save_json_gz(cluster_info_payload, output_paths["cluster_info"])
    save_json_gz(cluster_patterns_payload, output_paths["cluster_patterns"])

    print("[Session] saved session_info     :", output_paths["session_info"])
    print("[Session] saved cluster_info     :", output_paths["cluster_info"])
    print("[Session] saved cluster_patterns :", output_paths["cluster_patterns"])

    # --------------------------------------------------------
    # 6. Export K-dependent session pattern files
    # --------------------------------------------------------
    topk_files = export_topk_session_pattern_files(
        cfg,
        cluster_patterns,
    )

    for path in topk_files:
        print("[Session] saved topK pattern:", path)

    # --------------------------------------------------------
    # 7. Save report
    # --------------------------------------------------------
    report = {
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "config_summary": {
            "dataset": dataset_cfg["name"],
            "raw_log_path": str(raw_log_path),
            "gap_hours": gap_hours,
            "algorithm": algorithm,
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "k_values": cfg["abstraction"]["k_values"],
        },
        "outputs": {
            "cut_dir": str(output_paths["cut_dir"]),
            "session_info": str(output_paths["session_info"]),
            "cluster_info": str(output_paths["cluster_info"]),
            "cluster_patterns": str(output_paths["cluster_patterns"]),
            "topk_files": [str(p) for p in topk_files],
            "report": str(output_paths["report"]),
        },
        "summary": {
            "n_sessions": len(clustered_sessions),
            "vocab_size": len(vocab),
            "n_clusters": cluster_summary["n_clusters"],
            "n_noise": cluster_summary["n_noise"],
            "n_cluster_patterns": len(cluster_patterns),
        },
    }

    save_json(report, output_paths["report"])

    print("[Session] saved report:", output_paths["report"])
    print("=" * 80)
    print("[Session] Done")
    print("=" * 80)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build session-based pattern files."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/bpic2015.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_session_pipeline(cfg)


if __name__ == "__main__":
    main()