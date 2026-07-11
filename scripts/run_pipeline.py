from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bpic2015.yaml"
PIPELINE_STAGE_ORDER = [f"{i:02d}" for i in range(1, 8)]


# ============================================================
# Basic utilities
# ============================================================
def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.is_absolute():
        candidate_from_cwd = Path.cwd() / path
        candidate_from_root = PROJECT_ROOT / path

        if candidate_from_cwd.exists():
            path = candidate_from_cwd
        else:
            path = candidate_from_root

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config root must be a mapping: {path}"
        )

    cfg["_config_path"] = str(path)

    return cfg


def save_json(
    path: str | Path,
    obj: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return path


def utc_now_text() -> str:
    return datetime.now().isoformat(
        timespec="seconds"
    )


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:06.2f}"
    )


# ============================================================
# Config validation
# ============================================================
def validate_pipeline_config(
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    pipeline_cfg = cfg.get("pipeline")

    if not isinstance(pipeline_cfg, dict):
        raise KeyError(
            "Missing YAML section: pipeline"
        )

    stages = pipeline_cfg.get("stages")

    if not isinstance(stages, dict):
        raise KeyError(
            "Missing YAML section: pipeline.stages"
        )

    normalized: Dict[str, Dict[str, Any]] = {}

    for stage_id in PIPELINE_STAGE_ORDER:
        stage_cfg = stages.get(stage_id)

        if stage_cfg is None:
            raise KeyError(
                f"Missing pipeline stage configuration: "
                f"pipeline.stages.{stage_id}"
            )

        if not isinstance(stage_cfg, dict):
            raise TypeError(
                f"pipeline.stages.{stage_id} "
                "must be a mapping."
            )

        script_value = stage_cfg.get("script")

        if not script_value:
            raise KeyError(
                f"Missing script path for stage {stage_id}"
            )

        script_path = Path(str(script_value))

        if not script_path.is_absolute():
            script_path = PROJECT_ROOT / script_path

        script_path = script_path.resolve()

        if not script_path.exists():
            raise FileNotFoundError(
                f"Stage {stage_id} script does not exist: "
                f"{script_path}"
            )

        stage_args = stage_cfg.get("args", {})

        if stage_args is None:
            stage_args = {}

        if not isinstance(stage_args, dict):
            raise TypeError(
                f"pipeline.stages.{stage_id}.args "
                "must be a mapping."
            )

        normalized[stage_id] = {
            **stage_cfg,
            "enabled": bool(
                stage_cfg.get("enabled", True)
            ),
            "name": str(
                stage_cfg.get(
                    "name",
                    script_path.stem,
                )
            ),
            "script_path": script_path,
            "args": stage_args,
        }

    validate_shared_stage_values(normalized)

    return normalized


def validate_shared_stage_values(
    stages: Dict[str, Dict[str, Any]],
) -> None:
    """
    Check values that must remain identical across stages.
    """
    rank_values: Dict[str, int] = {}

    for stage_id in ("05", "06", "07"):
        args = stages[stage_id].get("args", {})

        if "rank" in args:
            rank_values[stage_id] = int(
                args["rank"]
            )

    if rank_values:
        unique_ranks = set(rank_values.values())

        if len(unique_ranks) != 1:
            raise ValueError(
                "Rank mismatch across stages 05, 06, and 07: "
                f"{rank_values}"
            )

    abstract_noise = (
        stages["05"]
        .get("args", {})
        .get("abstract-noise")
    )

    if abstract_noise is not None:
        value = float(abstract_noise)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Stage 05 abstract-noise must be "
                f"between 0 and 1, got {value}"
            )

    local_noise = (
        stages["05"]
        .get("args", {})
        .get("local-noise")
    )

    if local_noise is not None:
        value = float(local_noise)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Stage 05 local-noise must be "
                f"between 0 and 1, got {value}"
            )

    org_noise = (
        stages["06"]
        .get("args", {})
        .get("org-noise")
    )

    if org_noise is not None:
        value = float(org_noise)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Stage 06 org-noise must be "
                f"between 0 and 1, got {value}"
            )


# ============================================================
# CLI argument conversion
# ============================================================
def normalize_option_name(name: str) -> str:
    """
    YAML:
        progress_every
        progress-every

    Both become:
        --progress-every
    """
    normalized = str(name).strip().replace("_", "-")

    if normalized.startswith("--"):
        return normalized

    return f"--{normalized}"


def append_cli_argument(
    command: List[str],
    key: str,
    value: Any,
) -> None:
    """
    Convert YAML values into CLI arguments.

    Rules
    -----
    true:
        --flag

    false / null:
        omitted

    scalar:
        --key value

    list:
        --key value1 value2 ...
    """
    option = normalize_option_name(key)

    if value is None:
        return

    if isinstance(value, bool):
        if value:
            command.append(option)

        return

    if isinstance(value, (list, tuple)):
        if not value:
            return

        command.append(option)
        command.extend(str(item) for item in value)
        return

    command.extend(
        [
            option,
            str(value),
        ]
    )


def build_stage_command(
    *,
    stage_cfg: Dict[str, Any],
    config_path: Path,
) -> List[str]:
    command = [
        sys.executable,
        str(stage_cfg["script_path"]),
        "--config",
        str(config_path),
    ]

    for key, value in stage_cfg.get(
        "args",
        {},
    ).items():
        append_cli_argument(
            command,
            str(key),
            value,
        )

    return command


# ============================================================
# Stage selection
# ============================================================
def stage_in_range(
    stage_id: str,
    *,
    from_stage: str,
    to_stage: str,
) -> bool:
    return (
        int(from_stage)
        <= int(stage_id)
        <= int(to_stage)
    )


def normalize_stage_id(value: str | int) -> str:
    stage_id = f"{int(value):02d}"

    if stage_id not in PIPELINE_STAGE_ORDER:
        raise ValueError(
            f"Stage must be between 1 and 7, got {value}"
        )

    return stage_id


def select_stages(
    stages: Dict[str, Dict[str, Any]],
    *,
    from_stage: str,
    to_stage: str,
    skipped_stages: Iterable[str],
) -> List[str]:
    skipped = {
        normalize_stage_id(stage)
        for stage in skipped_stages
    }

    selected: List[str] = []

    for stage_id in PIPELINE_STAGE_ORDER:
        if not stage_in_range(
            stage_id,
            from_stage=from_stage,
            to_stage=to_stage,
        ):
            continue

        if stage_id in skipped:
            continue

        if not stages[stage_id]["enabled"]:
            continue

        selected.append(stage_id)

    return selected


# ============================================================
# Execution
# ============================================================
def build_subprocess_environment() -> Dict[str, str]:
    env = os.environ.copy()

    existing_pythonpath = env.get(
        "PYTHONPATH",
        "",
    )

    pythonpath_parts = [
        str(SRC_DIR),
    ]

    if existing_pythonpath:
        pythonpath_parts.append(
            existing_pythonpath
        )

    env["PYTHONPATH"] = os.pathsep.join(
        pythonpath_parts
    )

    env["PYTHONUNBUFFERED"] = "1"

    return env


def run_one_stage(
    *,
    stage_id: str,
    stage_cfg: Dict[str, Any],
    config_path: Path,
    log_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    command = build_stage_command(
        stage_cfg=stage_cfg,
        config_path=config_path,
    )

    command_text = shlex.join(command)

    log_path = (
        log_dir
        / f"stage_{stage_id}_{stage_cfg['name']}.log"
    )

    print()
    print("=" * 100)
    print(
        f"[RunAll] Stage {stage_id}: "
        f"{stage_cfg['name']}"
    )
    print("=" * 100)
    print(f"[RunAll] command: {command_text}")
    print(f"[RunAll] log    : {log_path}")

    started_at = utc_now_text()
    started_perf = time.perf_counter()

    if dry_run:
        print("[RunAll] DRY RUN: command not executed.")

        return {
            "stage": stage_id,
            "name": stage_cfg["name"],
            "status": "dry_run",
            "command": command,
            "command_text": command_text,
            "log_path": str(log_path),
            "started_at": started_at,
            "finished_at": utc_now_text(),
            "duration_seconds": 0.0,
            "return_code": None,
        }

    env = build_subprocess_environment()

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return_code: Optional[int] = None
    error_text: Optional[str] = None

    try:
        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                f"[RunAll] command: {command_text}\n"
            )
            log_file.flush()

            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()

    except Exception:
        error_text = traceback.format_exc()
        return_code = -1

        print(error_text, flush=True)

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write("\n")
            log_file.write(error_text)

    duration = time.perf_counter() - started_perf

    status = (
        "completed"
        if return_code == 0
        else "failed"
    )

    print(
        f"[RunAll] Stage {stage_id} {status} "
        f"| duration={format_duration(duration)} "
        f"| return_code={return_code}"
    )

    return {
        "stage": stage_id,
        "name": stage_cfg["name"],
        "status": status,
        "command": command,
        "command_text": command_text,
        "log_path": str(log_path),
        "started_at": started_at,
        "finished_at": utc_now_text(),
        "duration_seconds": float(duration),
        "return_code": return_code,
        "error": error_text,
    }


def run_pipeline(
    *,
    config_path: Path,
    from_stage: str,
    to_stage: str,
    skipped_stages: Iterable[str],
    dry_run: bool,
    continue_on_error: bool,
) -> Dict[str, Any]:
    cfg = load_yaml(config_path)

    resolved_config_path = Path(
        cfg["_config_path"]
    ).resolve()

    stages = validate_pipeline_config(cfg)

    selected_stage_ids = select_stages(
        stages,
        from_stage=from_stage,
        to_stage=to_stage,
        skipped_stages=skipped_stages,
    )

    if not selected_stage_ids:
        raise ValueError(
            "No enabled stages were selected."
        )

    dataset_name = str(
        cfg.get(
            "dataset",
            {},
        ).get(
            "name",
            "unknown_dataset",
        )
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    pipeline_run_dir = (
        PROJECT_ROOT
        / "results"
        / "pipeline_runs"
        / dataset_name
        / timestamp
    )

    log_dir = pipeline_run_dir / "logs"

    pipeline_run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        pipeline_run_dir
        / "pipeline_report.json"
    )

    pipeline_cfg = cfg.get(
        "pipeline",
        {},
    )

    yaml_stop_on_error = bool(
        pipeline_cfg.get(
            "stop_on_error",
            True,
        )
    )

    effective_stop_on_error = (
        yaml_stop_on_error
        and not continue_on_error
    )

    print("=" * 100)
    print("[RunAll] Event Abstraction Ensemble Pipeline")
    print("=" * 100)
    print(f"[RunAll] project root : {PROJECT_ROOT}")
    print(f"[RunAll] config       : {resolved_config_path}")
    print(f"[RunAll] dataset      : {dataset_name}")
    print(
        f"[RunAll] stages       : "
        f"{', '.join(selected_stage_ids)}"
    )
    print(
        f"[RunAll] stop on error: "
        f"{effective_stop_on_error}"
    )
    print(f"[RunAll] dry run      : {dry_run}")
    print(f"[RunAll] run directory: {pipeline_run_dir}")

    pipeline_started_at = utc_now_text()
    pipeline_started_perf = time.perf_counter()

    stage_results: List[Dict[str, Any]] = []
    pipeline_status = "completed"

    for stage_id in selected_stage_ids:
        stage_result = run_one_stage(
            stage_id=stage_id,
            stage_cfg=stages[stage_id],
            config_path=resolved_config_path,
            log_dir=log_dir,
            dry_run=dry_run,
        )

        stage_results.append(stage_result)

        partial_report = {
            "status": "running",
            "dataset": dataset_name,
            "config_path": str(
                resolved_config_path
            ),
            "project_root": str(PROJECT_ROOT),
            "started_at": pipeline_started_at,
            "last_updated_at": utc_now_text(),
            "selected_stages": selected_stage_ids,
            "dry_run": bool(dry_run),
            "stop_on_error": bool(
                effective_stop_on_error
            ),
            "stage_results": stage_results,
        }

        save_json(
            report_path,
            partial_report,
        )

        if stage_result["status"] == "failed":
            pipeline_status = "failed"

            if effective_stop_on_error:
                print(
                    f"[RunAll] Stopping after failed "
                    f"stage {stage_id}."
                )
                break

    total_duration = (
        time.perf_counter()
        - pipeline_started_perf
    )

    n_completed = sum(
        result["status"] == "completed"
        for result in stage_results
    )

    n_failed = sum(
        result["status"] == "failed"
        for result in stage_results
    )

    if dry_run:
        pipeline_status = "dry_run"
    elif n_failed > 0:
        pipeline_status = "failed"

    final_report = {
        "status": pipeline_status,
        "dataset": dataset_name,
        "config_path": str(
            resolved_config_path
        ),
        "project_root": str(PROJECT_ROOT),
        "pipeline_run_dir": str(
            pipeline_run_dir
        ),
        "started_at": pipeline_started_at,
        "finished_at": utc_now_text(),
        "duration_seconds": float(
            total_duration
        ),
        "duration_text": format_duration(
            total_duration
        ),
        "selected_stages": selected_stage_ids,
        "dry_run": bool(dry_run),
        "stop_on_error": bool(
            effective_stop_on_error
        ),
        "summary": {
            "n_selected": len(
                selected_stage_ids
            ),
            "n_executed": len(
                stage_results
            ),
            "n_completed": int(
                n_completed
            ),
            "n_failed": int(n_failed),
        },
        "stage_results": stage_results,
    }

    save_json(
        report_path,
        final_report,
    )

    print()
    print("=" * 100)
    print("[RunAll] Pipeline summary")
    print("=" * 100)
    print(
        f"[RunAll] status   : "
        f"{pipeline_status}"
    )
    print(
        f"[RunAll] completed: "
        f"{n_completed}"
    )
    print(
        f"[RunAll] failed   : "
        f"{n_failed}"
    )
    print(
        f"[RunAll] duration : "
        f"{format_duration(total_duration)}"
    )
    print(
        f"[RunAll] report   : "
        f"{report_path}"
    )

    if pipeline_status == "failed":
        raise SystemExit(1)

    return final_report


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run stages 01 through 07 using one YAML config."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=(
            "Path to the dataset YAML configuration."
        ),
    )

    parser.add_argument(
        "--from-stage",
        type=str,
        default="01",
        help="First stage to execute. Default: 01.",
    )

    parser.add_argument(
        "--to-stage",
        type=str,
        default="07",
        help="Last stage to execute. Default: 07.",
    )

    parser.add_argument(
        "--skip-stage",
        action="append",
        default=[],
        help=(
            "Stage to skip. Can be specified multiple times, "
            "for example: --skip-stage 01 --skip-stage 02"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print and validate commands without executing them."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue with later stages after a failure. "
            "Normally not recommended because stages depend "
            "on previous outputs."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from_stage = normalize_stage_id(
        args.from_stage
    )
    to_stage = normalize_stage_id(
        args.to_stage
    )

    if int(from_stage) > int(to_stage):
        raise ValueError(
            f"from-stage ({from_stage}) must not be "
            f"greater than to-stage ({to_stage})."
        )

    run_pipeline(
        config_path=Path(args.config),
        from_stage=from_stage,
        to_stage=to_stage,
        skipped_stages=args.skip_stage,
        dry_run=bool(args.dry_run),
        continue_on_error=bool(
            args.continue_on_error
        ),
    )


if __name__ == "__main__":
    main()