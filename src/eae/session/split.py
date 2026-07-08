# src/eae/session/split.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pm4py


@dataclass
class SessionRecord:
    session_id: int
    case_id: str
    session_index_in_case: int
    activities: List[str]
    timestamps: List[str]
    start_time: str | None
    end_time: str | None
    length: int


def load_event_log_dataframe(
    log_path: str | Path,
) -> pd.DataFrame:
    """
    Load an event log file as a pandas DataFrame.

    Supported mainly through PM4Py:
    - .xes
    - .xes.gz
    - other formats supported by pm4py.read_xes when applicable
    """
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Event log does not exist: {log_path}")

    suffixes = "".join(log_path.suffixes).lower()

    if suffixes.endswith(".xes") or suffixes.endswith(".xes.gz"):
        log = pm4py.read_xes(str(log_path))
        return pm4py.convert_to_dataframe(log)

    if suffixes.endswith(".csv"):
        return pd.read_csv(log_path)

    raise ValueError(f"Unsupported log format: {log_path}")


def normalize_event_dataframe(
    df: pd.DataFrame,
    *,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """
    Normalize the event log DataFrame.

    Returns columns:
    - case_id
    - activity
    - timestamp
    """
    missing = [
        col
        for col in [case_col, activity_col, timestamp_col]
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns in event log: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[case_col, activity_col, timestamp_col]].copy()
    out.columns = ["case_id", "activity", "timestamp"]

    out["case_id"] = out["case_id"].astype(str)
    out["activity"] = out["activity"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    out = out.dropna(subset=["timestamp"])
    out = out.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    return out


def split_sessions_by_time_gap(
    df: pd.DataFrame,
    *,
    gap_hours: float,
) -> List[Dict[str, Any]]:
    """
    Split each case into sessions using a time-gap threshold.

    A new session starts when:
        current_timestamp - previous_timestamp > gap_hours
    within the same case.
    """
    if gap_hours <= 0:
        raise ValueError(f"gap_hours must be positive, got {gap_hours}")

    gap = pd.Timedelta(hours=float(gap_hours))

    sessions: List[Dict[str, Any]] = []
    global_session_id = 0

    for case_id, g in df.groupby("case_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        current_activities: List[str] = []
        current_timestamps: List[pd.Timestamp] = []
        session_idx = 0
        prev_ts = None

        def flush_session() -> None:
            nonlocal global_session_id, session_idx
            if not current_activities:
                return

            start_ts = current_timestamps[0]
            end_ts = current_timestamps[-1]

            sessions.append(
                {
                    "session_id": int(global_session_id),
                    "case_id": str(case_id),
                    "session_index_in_case": int(session_idx),
                    "activities": list(current_activities),
                    "timestamps": [ts.isoformat() for ts in current_timestamps],
                    "start_time": start_ts.isoformat() if start_ts is not None else None,
                    "end_time": end_ts.isoformat() if end_ts is not None else None,
                    "length": int(len(current_activities)),
                }
            )

            global_session_id += 1
            session_idx += 1

        for _, row in g.iterrows():
            ts = row["timestamp"]
            act = row["activity"]

            if prev_ts is not None and ts - prev_ts > gap:
                flush_session()
                current_activities = []
                current_timestamps = []

            current_activities.append(str(act))
            current_timestamps.append(ts)
            prev_ts = ts

        flush_session()

    return sessions


def build_sessions_from_log(
    log_path: str | Path,
    *,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
    gap_hours: float,
) -> List[Dict[str, Any]]:
    """
    Full session-splitting helper:
    raw log -> normalized dataframe -> session records.
    """
    df_raw = load_event_log_dataframe(log_path)
    df = normalize_event_dataframe(
        df_raw,
        case_col=case_col,
        activity_col=activity_col,
        timestamp_col=timestamp_col,
    )

    return split_sessions_by_time_gap(
        df,
        gap_hours=gap_hours,
    )