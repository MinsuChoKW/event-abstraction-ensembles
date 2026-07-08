# src/eae/session/cluster.py

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_sessions_dbscan(
    X: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Cluster session vectors using DBSCAN.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    if min_samples <= 0:
        raise ValueError(f"min_samples must be positive, got {min_samples}")

    model = DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric=metric,
    )

    labels = model.fit_predict(X)
    return labels


def attach_cluster_labels(
    sessions: List[Dict[str, Any]],
    labels: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Attach DBSCAN cluster labels to session records.
    """
    if len(sessions) != len(labels):
        raise ValueError(
            f"len(sessions) != len(labels): {len(sessions)} != {len(labels)}"
        )

    out = []

    for session, label in zip(sessions, labels):
        item = dict(session)
        item["cluster_label"] = int(label)
        item["is_noise"] = bool(int(label) == -1)
        out.append(item)

    return out


def summarize_clusters(
    clustered_sessions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarize DBSCAN cluster labels.
    """
    label_counts = Counter(int(s["cluster_label"]) for s in clustered_sessions)

    non_noise = {
        label: count
        for label, count in label_counts.items()
        if label != -1
    }

    return {
        "n_sessions": len(clustered_sessions),
        "n_clusters": len(non_noise),
        "n_noise": int(label_counts.get(-1, 0)),
        "cluster_sizes": dict(sorted(label_counts.items(), key=lambda x: x[0])),
    }


def build_cluster_pattern_table(
    clustered_sessions: List[Dict[str, Any]],
    *,
    include_noise: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build a ranked table of distinct session sequences inside clusters.

    Each row corresponds to one distinct sequence in one cluster.
    """
    bucket = defaultdict(list)

    for s in clustered_sessions:
        label = int(s["cluster_label"])

        if label == -1 and not include_noise:
            continue

        sequence = tuple(str(a) for a in s.get("sequence", s.get("activities", [])))
        bucket[(label, sequence)].append(s)

    cluster_size = Counter(
        int(s["cluster_label"])
        for s in clustered_sessions
        if include_noise or int(s["cluster_label"]) != -1
    )

    rows = []

    for (label, sequence), members in bucket.items():
        support = len(members)

        rows.append(
            {
                "cluster_label": int(label),
                "cluster_size": int(cluster_size[label]),
                "support": int(support),
                "sequence": list(sequence),
                "sequence_str": " ".join(sequence),
                "length": int(len(sequence)),
                "example_session_id": int(members[0]["session_id"]),
                "example_case_id": str(members[0]["case_id"]),
            }
        )

    rows.sort(
        key=lambda r: (
            -r["support"],
            -r["cluster_size"],
            r["cluster_label"],
            r["length"],
            r["sequence_str"],
        )
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)

    return rows