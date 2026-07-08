# src/eae/evaluation/model_io.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pm4py

from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet


def load_xes_log(
    path: str | Path,
) -> EventLog:
    """
    Load an XES event log.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"XES log does not exist: {path}"
        )

    try:
        return pm4py.read_xes(
            str(path),
            return_legacy_log_object=True,
        )
    except TypeError:
        return pm4py.read_xes(str(path))


def load_petri_net(
    path: str | Path,
) -> Tuple[PetriNet, Marking, Marking]:
    """
    Load an accepting Petri net from PNML.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"PNML model does not exist: {path}"
        )

    try:
        return pm4py.read_pnml(str(path))
    except Exception:
        from pm4py.objects.petri_net.importer import (
            importer as pnml_importer,
        )

        return pnml_importer.apply(str(path))


def summarize_event_log(
    log: EventLog,
) -> Dict[str, Any]:
    """
    Return basic event-log statistics.
    """
    lengths = [
        len(trace)
        for trace in log
    ]

    activities = set()

    for trace in log:
        for event in trace:
            activity = event.get("concept:name")

            if activity is not None:
                activities.add(str(activity))

    return {
        "n_cases": int(len(log)),
        "n_events": int(sum(lengths)),
        "n_activities": int(len(activities)),
        "min_case_len": (
            int(min(lengths))
            if lengths
            else 0
        ),
        "max_case_len": (
            int(max(lengths))
            if lengths
            else 0
        ),
        "mean_case_len": (
            float(sum(lengths) / len(lengths))
            if lengths
            else 0.0
        ),
    }


def summarize_petri_net(
    net: PetriNet,
) -> Dict[str, Any]:
    """
    Return basic Petri-net statistics.
    """
    visible = [
        transition
        for transition in net.transitions
        if transition.label is not None
    ]

    silent = [
        transition
        for transition in net.transitions
        if transition.label is None
    ]

    return {
        "n_places": int(len(net.places)),
        "n_transitions": int(
            len(net.transitions)
        ),
        "n_visible_transitions": int(
            len(visible)
        ),
        "n_silent_transitions": int(
            len(silent)
        ),
        "n_arcs": int(len(net.arcs)),
        "visible_labels": sorted(
            str(transition.label)
            for transition in visible
        ),
    }