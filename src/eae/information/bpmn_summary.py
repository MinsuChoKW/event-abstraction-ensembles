from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd


BPMN_ACTIVITY_TAGS = {
    "task",
    "userTask",
    "manualTask",
    "serviceTask",
    "scriptTask",
    "businessRuleTask",
    "sendTask",
    "receiveTask",
    "callActivity",
    "subProcess",
    "transaction",
    "adHocSubProcess",
}

BPMN_EVENT_TAGS = {
    "startEvent",
    "intermediateCatchEvent",
    "intermediateThrowEvent",
    "boundaryEvent",
    "endEvent",
}

BPMN_GATEWAY_TAGS = {
    "exclusiveGateway",
    "parallelGateway",
    "inclusiveGateway",
    "eventBasedGateway",
    "complexGateway",
}

BPMN_FLOW_NODE_TAGS = (
    BPMN_ACTIVITY_TAGS | BPMN_EVENT_TAGS | BPMN_GATEWAY_TAGS
)


def _xml_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_bpmn_size(bpmn_path: str | Path) -> dict[str, int]:
    bpmn_path = Path(bpmn_path)
    if not bpmn_path.exists():
        raise FileNotFoundError(f"BPMN file does not exist: {bpmn_path}")

    try:
        tree = ET.parse(bpmn_path)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid BPMN XML file: {bpmn_path}") from exc

    counts = Counter(
        _xml_local_name(element.tag)
        for element in tree.getroot().iter()
    )

    return {
        "Model activities": int(
            sum(counts[tag] for tag in BPMN_ACTIVITY_TAGS)
        ),
        "Start/end events": int(
            counts["startEvent"] + counts["endEvent"]
        ),
        "XOR gateways": int(counts["exclusiveGateway"]),
        "AND gateways": int(counts["parallelGateway"]),
        "Total gateways": int(
            sum(counts[tag] for tag in BPMN_GATEWAY_TAGS)
        ),
        "Sequence flows": int(counts["sequenceFlow"]),
        "Total flow nodes": int(
            sum(counts[tag] for tag in BPMN_FLOW_NODE_TAGS)
        ),
    }


def _format_percentage_change(original: int, abstracted: int) -> str:
    if original == 0:
        return "0" if abstracted == 0 else f"{abstracted:,} (n/a)"

    change = 100.0 * (abstracted - original) / original
    rounded_change = int(round(change))
    if rounded_change == 0:
        return f"{abstracted:,}"
    return f"{abstracted:,} ({rounded_change:+d}%)"


def resolve_bpmn_paths(
    *,
    run_dir: Path,
    rank: int,
) -> tuple[Path, Path]:
    """Resolve BPMN files produced by Stage 07."""
    complexity_dir = run_dir / "evaluation" / "complexity"

    original_bpmn_path = complexity_dir / "ORG_model.bpmn"
    abstracted_bpmn_path = (
        complexity_dir / f"ABS_model_rank{rank}.bpmn"
    )

    missing = [
        path
        for path in (original_bpmn_path, abstracted_bpmn_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required BPMN files are missing. "
            "Run Stage 07 for this experiment first:\n"
            + "\n".join(str(path) for path in missing)
        )

    return original_bpmn_path, abstracted_bpmn_path


def build_bpmn_size_summary(
    original_bpmn_path: str | Path,
    abstracted_bpmn_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build numeric and formatted ORG-vs-ABS BPMN size tables."""
    original_stats = _parse_bpmn_size(original_bpmn_path)
    abstracted_stats = _parse_bpmn_size(abstracted_bpmn_path)

    metric_order = [
        "Model activities",
        "Start/end events",
        "XOR gateways",
        "AND gateways",
        "Total gateways",
        "Sequence flows",
        "Total flow nodes",
    ]

    numeric_table = pd.DataFrame(
        {"ORG": original_stats, "ABS": abstracted_stats}
    ).loc[metric_order]

    display_table = pd.DataFrame(
        index=metric_order,
        columns=["ORG", "ABS"],
        dtype=object,
    )

    for metric in metric_order:
        original_value = int(numeric_table.loc[metric, "ORG"])
        abstracted_value = int(numeric_table.loc[metric, "ABS"])

        display_table.loc[metric, "ORG"] = f"{original_value:,}"
        display_table.loc[metric, "ABS"] = _format_percentage_change(
            original_value,
            abstracted_value,
        )

    return numeric_table, display_table


def save_bpmn_size_summary(
    numeric_table: pd.DataFrame,
    display_table: pd.DataFrame,
    *,
    output_dir: str | Path,
    output_stem: str,
) -> dict[str, Path]:
    """Save numeric and formatted BPMN size tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_csv_path = output_dir / f"{output_stem}.csv"
    formatted_csv_path = output_dir / f"{output_stem}_formatted.csv"

    numeric_table.to_csv(
        numeric_csv_path,
        index=True,
        index_label="Metric",
    )
    display_table.to_csv(
        formatted_csv_path,
        index=True,
        index_label="Metric",
    )

    return {
        "numeric_csv": numeric_csv_path,
        "formatted_csv": formatted_csv_path,
    }
