# src/eae/evaluation/complexity.py

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pm4py


UNDERSTANDBPMN_METRIC_NAMES = [
    "coef_network_connectivity",
    "cognitive_weight",
    "connector_heterogeneity",
    "control_flow_complexity",
    "density",
    "sequentiality",
    "token_split",
]


R_METRICS_SCRIPT = r"""
suppressPackageStartupMessages(library(understandBPMN))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
    stop("BPMN file path is required.")
}

f <- args[1]

out <- list(
    coef_network_connectivity = coefficient_network_connectivity(f),
    cognitive_weight          = cognitive_weight(f),
    connector_heterogeneity   = connector_heterogeneity(f),
    control_flow_complexity   = control_flow_complexity(f),
    density                   = density_process_model(f),
    sequentiality             = sequentiality(f),
    token_split               = token_split(f)
)

cat(toJSON(out, auto_unbox = TRUE, na = "null"))
"""


def verify_understandbpmn_ready() -> Dict[str, Any]:
    """
    Verify that Rscript, jsonlite, and understandBPMN are installed.

    Installation itself is handled by:
      scripts/install_understandbpmn.sh
    """
    checks = {
        "rscript": False,
        "jsonlite": False,
        "understandBPMN": False,
    }

    try:
        subprocess.run(
            ["Rscript", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        checks["rscript"] = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    if not checks["rscript"]:
        raise RuntimeError(
            "Rscript is not available.\n"
            "Run:\n"
            "  bash scripts/install_understandbpmn.sh"
        )

    verification_script = """
packages <- c("jsonlite", "understandBPMN")

missing <- packages[
    !vapply(
        packages,
        requireNamespace,
        logical(1),
        quietly = TRUE
    )
]

if (length(missing) > 0) {
    cat(paste(missing, collapse=","))
    quit(status=1)
}

cat("OK")
"""

    proc = subprocess.run(
        ["Rscript", "-e", verification_script],
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Required R packages are missing.\n"
            f"R output: {proc.stdout}\n"
            f"R error : {proc.stderr}\n"
            "Run:\n"
            "  bash scripts/install_understandbpmn.sh"
        )

    checks["jsonlite"] = True
    checks["understandBPMN"] = True

    print("[Complexity] Rscript       : OK")
    print("[Complexity] jsonlite      : OK")
    print("[Complexity] understandBPMN: OK")

    return checks


def load_process_tree(
    ptml_path: str | Path,
):
    """
    Load a process tree from PTML.
    """
    ptml_path = Path(ptml_path)

    if not ptml_path.exists():
        raise FileNotFoundError(
            f"PTML file does not exist: {ptml_path}"
        )

    try:
        return pm4py.read_ptml(str(ptml_path))

    except Exception:
        from pm4py.objects.process_tree.importer import (
            importer as process_tree_importer,
        )

        return process_tree_importer.apply(
            str(ptml_path)
        )


def process_tree_to_bpmn(
    tree,
):
    """
    Convert a PM4Py process tree to a BPMN model.
    """
    try:
        return pm4py.convert_to_bpmn(tree)

    except Exception:
        from pm4py.objects.conversion.process_tree import (
            converter as process_tree_converter,
        )

        try:
            return process_tree_converter.apply(
                tree,
                variant=(
                    process_tree_converter
                    .Variants
                    .TO_BPMN
                ),
            )

        except Exception:
            from pm4py.objects.conversion.process_tree.variants import (
                to_bpmn,
            )

            return to_bpmn.apply(tree)


def write_bpmn(
    bpmn,
    bpmn_path: str | Path,
) -> Path:
    """
    Export a BPMN model as BPMN XML.
    """
    bpmn_path = Path(bpmn_path)
    bpmn_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        pm4py.write_bpmn(
            bpmn,
            str(bpmn_path),
        )

    except Exception:
        from pm4py.objects.bpmn.exporter import (
            exporter as bpmn_exporter,
        )

        bpmn_exporter.apply(
            bpmn,
            str(bpmn_path),
        )

    if not bpmn_path.exists():
        raise RuntimeError(
            f"BPMN export failed: {bpmn_path}"
        )

    return bpmn_path


def convert_ptml_to_bpmn(
    ptml_path: str | Path,
    bpmn_path: str | Path,
) -> Path:
    """
    Load a PTML process tree and export the corresponding BPMN model.
    """
    tree = load_process_tree(ptml_path)
    bpmn = process_tree_to_bpmn(tree)

    return write_bpmn(
        bpmn,
        bpmn_path,
    )


def _safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None

        value = value[0]

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(result) or math.isinf(result):
        return None

    return result


def understandbpmn_metrics(
    bpmn_path: str | Path,
) -> Dict[str, float | None]:
    """
    Compute the seven original understandBPMN metrics.

    Metrics:
      - coefficient of network connectivity
      - cognitive weight
      - connector heterogeneity
      - control-flow complexity
      - density
      - sequentiality
      - token split
    """
    bpmn_path = Path(bpmn_path)

    if not bpmn_path.exists():
        raise FileNotFoundError(
            f"BPMN file does not exist: {bpmn_path}"
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".R",
        encoding="utf-8",
        delete=False,
    ) as temp_file:
        temp_file.write(R_METRICS_SCRIPT)
        r_script_path = Path(temp_file.name)

    try:
        proc = subprocess.run(
            [
                "Rscript",
                str(r_script_path),
                str(bpmn_path),
            ],
            text=True,
            capture_output=True,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "understandBPMN execution failed.\n"
                f"BPMN: {bpmn_path}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        stdout = proc.stdout.strip()

        if not stdout:
            raise RuntimeError(
                "understandBPMN returned empty output.\n"
                f"BPMN: {bpmn_path}"
            )

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Failed to parse understandBPMN JSON output.\n"
                f"Output:\n{stdout}"
            ) from exc

        return {
            metric_name: _safe_float(
                payload.get(metric_name)
            )
            for metric_name
            in UNDERSTANDBPMN_METRIC_NAMES
        }

    finally:
        try:
            os.remove(r_script_path)
        except OSError:
            pass


def convert_bpmn_to_petri_net(
    bpmn_path: str | Path,
):
    """
    Load BPMN and convert it to a Petri net.
    """
    bpmn_path = Path(bpmn_path)

    if not bpmn_path.exists():
        raise FileNotFoundError(
            f"BPMN file does not exist: {bpmn_path}"
        )

    bpmn = pm4py.read_bpmn(
        str(bpmn_path)
    )

    try:
        return pm4py.convert_to_petri_net(
            bpmn
        )

    except Exception:
        from pm4py.objects.conversion.bpmn import (
            converter as bpmn_converter,
        )

        return bpmn_converter.apply(
            bpmn
        )


def inverse_arc_degree_from_petri_net(
    net,
    initial_marking,
    final_marking,
) -> float:
    """
    Compute PM4Py arc-degree simplicity.

    A manual fallback reproduces the old notebook fallback:
      1 / (1 + mean node degree)
    """
    try:
        from pm4py.algo.evaluation.simplicity import (
            algorithm as simplicity_algorithm,
        )

        value = simplicity_algorithm.apply(
            net,
            initial_marking,
            final_marking,
            variant=(
                simplicity_algorithm
                .Variants
                .ARC_DEGREE
            ),
        )

        return float(value)

    except Exception:
        nodes = (
            list(net.places)
            + list(net.transitions)
        )

        if not nodes:
            return 1.0

        degrees = []

        for node in nodes:
            in_degree = len(
                getattr(
                    node,
                    "in_arcs",
                    [],
                )
            )

            out_degree = len(
                getattr(
                    node,
                    "out_arcs",
                    [],
                )
            )

            degrees.append(
                in_degree + out_degree
            )

        mean_degree = (
            sum(degrees) / len(degrees)
        )

        return float(
            1.0 / (1.0 + mean_degree)
        )


def pm4py_inverse_arc_degree(
    bpmn_path: str | Path,
) -> float:
    """
    Compute inverse arc degree on the Petri net converted from BPMN.
    """
    net, initial_marking, final_marking = (
        convert_bpmn_to_petri_net(
            bpmn_path
        )
    )

    return inverse_arc_degree_from_petri_net(
        net,
        initial_marking,
        final_marking,
    )


def compute_bpmn_complexity_metrics(
    bpmn_path: str | Path,
) -> Dict[str, Any]:
    """
    Compute all eight complexity metrics for one BPMN model.
    """
    bpmn_path = Path(bpmn_path)

    r_metrics = understandbpmn_metrics(
        bpmn_path
    )

    inverse_arc_degree = (
        pm4py_inverse_arc_degree(
            bpmn_path
        )
    )

    return {
        "bpmn_path": str(bpmn_path),
        **r_metrics,
        "inverse_arc_degree": float(
            inverse_arc_degree
        ),
    }