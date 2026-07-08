# src/eae/session/encode.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np


def build_activity_vocabulary(
    sessions: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Build activity -> column index vocabulary from session records.
    """
    acts = sorted(
        {
            str(act)
            for session in sessions
            for act in session.get("activities", [])
        }
    )

    return {act: i for i, act in enumerate(acts)}


def encode_sessions_bow(
    sessions: List[Dict[str, Any]],
    *,
    normalize: str = "l1",
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode sessions as bag-of-activities vectors.

    Parameters
    ----------
    sessions
        List of session records.

    normalize
        - "none": raw counts
        - "l1"  : divide by total count per session
    """
    vocab = build_activity_vocabulary(sessions)
    X = np.zeros((len(sessions), len(vocab)), dtype=float)

    for i, session in enumerate(sessions):
        counts = Counter(session.get("activities", []))

        for act, cnt in counts.items():
            if act in vocab:
                X[i, vocab[act]] = float(cnt)

    if normalize == "l1":
        row_sum = X.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        X = X / row_sum
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unsupported normalize option: {normalize}")

    return X, vocab


def attach_session_sequences(
    sessions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add sequence_str to each session record.

    sequence_str is used for counting distinct session paths.
    """
    out = []

    for session in sessions:
        item = dict(session)
        acts = [str(a) for a in item.get("activities", [])]
        item["sequence"] = acts
        item["sequence_str"] = " ".join(acts)
        out.append(item)

    return out