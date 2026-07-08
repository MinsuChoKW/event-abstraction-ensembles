# src/eae/evaluation/conformance.py

from __future__ import annotations

import math
from typing import Any, Dict

from pm4py.algo.evaluation.precision import (
    algorithm as precision_evaluator,
)
from pm4py.algo.evaluation.replay_fitness import (
    algorithm as replay_fitness_evaluator,
)
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet


def harmonic_mean(
    precision: float,
    fitness: float,
) -> float:
    """
    Compute F1 as the harmonic mean of precision and fitness.
    """
    precision = float(precision)
    fitness = float(fitness)

    denominator = precision + fitness

    if denominator <= 0.0:
        return 0.0

    return float(
        2.0
        * precision
        * fitness
        / denominator
    )


def _safe_float(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)

    if math.isnan(value) or math.isinf(value):
        return float(default)

    return value


def compute_token_based_fitness(
    log: EventLog,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> Dict[str, Any]:
    """
    Compute token-based replay fitness.

    This follows the original notebook evaluation setting.
    """
    result = replay_fitness_evaluator.apply(
        log,
        net,
        initial_marking,
        final_marking,
        variant=(
            replay_fitness_evaluator
            .Variants
            .TOKEN_BASED
        ),
    )

    average_trace_fitness = _safe_float(
        result.get(
            "average_trace_fitness",
            result.get("log_fitness", 0.0),
        )
    )

    log_fitness = _safe_float(
        result.get(
            "log_fitness",
            average_trace_fitness,
        )
    )

    percentage_fit_traces = _safe_float(
        result.get(
            "percentage_of_fitting_traces",
            result.get(
                "perc_fit_traces",
                0.0,
            ),
        )
    )

    return {
        "average_trace_fitness": (
            average_trace_fitness
        ),
        "log_fitness": log_fitness,
        "percentage_of_fitting_traces": (
            percentage_fit_traces
        ),
        "raw_result": {
            str(key): (
                float(value)
                if isinstance(
                    value,
                    (int, float),
                )
                else value
            )
            for key, value in result.items()
        },
    }


def compute_etconformance_precision(
    log: EventLog,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> float:
    """
    Compute token-based ET-conformance precision.
    """
    try:
        precision = precision_evaluator.apply(
            log,
            net,
            initial_marking,
            final_marking,
            variant=(
                precision_evaluator
                .Variants
                .ETCONFORMANCE_TOKEN
            ),
        )

    except AttributeError:
        # Compatibility fallback for PM4Py versions
        # exposing only the lower-level token-based variant.
        from pm4py.algo.evaluation.precision.variants import (
            etconformance_token,
        )

        precision = etconformance_token.apply(
            log,
            net,
            initial_marking,
            final_marking,
        )

    return _safe_float(precision)


def evaluate_precision_fitness_f1(
    log: EventLog,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    *,
    label: str,
) -> Dict[str, Any]:
    """
    Original notebook-style conformance evaluation.

    Metrics
    -------
    recall_fitness
        Token-based replay fitness.

    precision
        ET-conformance token precision.

    f1
        Harmonic mean of replay fitness and precision.
    """
    print("-" * 80)
    print(f"[Conformance] {label}")
    print("-" * 80)

    print(
        "[Conformance] Computing "
        "token-based replay fitness",
        flush=True,
    )

    fitness_result = (
        compute_token_based_fitness(
            log,
            net,
            initial_marking,
            final_marking,
        )
    )

    recall_fitness = float(
        fitness_result[
            "average_trace_fitness"
        ]
    )

    print(
        "[Conformance] Computing "
        "ET-conformance precision",
        flush=True,
    )

    precision = (
        compute_etconformance_precision(
            log,
            net,
            initial_marking,
            final_marking,
        )
    )

    f1 = harmonic_mean(
        precision,
        recall_fitness,
    )

    result = {
        "label": str(label),

        # Names retained from the original notebook.
        "precision": float(precision),
        "recall_fitness": float(
            recall_fitness
        ),
        "f1": float(f1),

        "log_fitness": float(
            fitness_result["log_fitness"]
        ),
        "percentage_of_fitting_traces": (
            float(
                fitness_result[
                    "percentage_of_fitting_traces"
                ]
            )
        ),

        "fitness_method": (
            "token_based_replay"
        ),
        "precision_method": (
            "etconformance_token"
        ),
    }

    print(
        "[Conformance] precision      :",
        f"{precision:.10f}",
    )
    print(
        "[Conformance] replay fitness :",
        f"{recall_fitness:.10f}",
    )
    print(
        "[Conformance] F1             :",
        f"{f1:.10f}",
    )

    return result