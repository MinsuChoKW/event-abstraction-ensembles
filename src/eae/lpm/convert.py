# src/eae/lpm/convert.py

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict

from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
from pm4py.objects.process_tree.exporter import exporter as ptml_exporter


def extract_lpm_id_from_filename(path: str | Path) -> int | None:
    """
    Extract LPM id from filenames such as:

      lpm.1.3902696586630035445.pnml
      lpm.100.307971798554248492.pnml
      lpm.1.ptml

    Returns
    -------
    int | None
    """
    path = Path(path)

    m = re.search(
        r"lpm\.(\d+)(?:\.|$)",
        path.name,
        flags=re.IGNORECASE,
    )

    if not m:
        return None

    return int(m.group(1))


def convert_pnml_file_to_ptml(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Convert one PNML Petri net file to one PTML process-tree file.

    Parameters
    ----------
    input_path
        Source PNML file.

    output_path
        Target PTML file.

    overwrite
        If False and output already exists, conversion is skipped.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"PNML file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "reason": "output_exists",
            "input_path": str(input_path),
            "output_path": str(output_path),
        }

    try:
        net, initial_marking, final_marking = pnml_importer.apply(str(input_path))

        process_tree = wf_net_converter.apply(
            net,
            initial_marking,
            final_marking,
        )

        ptml_exporter.apply(
            process_tree,
            str(output_path),
        )

        return {
            "status": "converted",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "error": None,
        }

    except Exception as e:
        return {
            "status": "error",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "error": repr(e),
        }


def convert_pnml_dir_to_ptml(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Convert all PNML files in a directory to PTML files.

    Input example
    -------------
    data/bpic2015/raw/LPM_PN_Set/
      lpm.1.3902696586630035445.pnml
      lpm.2.12345.pnml

    Temporary output example
    ------------------------
    data/bpic2015/interim/LPM_PT_ver/
      lpm.1.3902696586630035445.ptml
      lpm.2.12345.ptml

    Canonical renaming is handled later by:
      rename_ptml_files_to_canonical()
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"PNML input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"PNML input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pnml_files = sorted(input_dir.glob("*.pnml"))

    converted = []
    skipped = []
    errors = []

    for pnml_path in pnml_files:
        output_path = output_dir / f"{pnml_path.stem}.ptml"

        report = convert_pnml_file_to_ptml(
            pnml_path,
            output_path,
            overwrite=overwrite,
        )

        if report["status"] == "converted":
            converted.append(report)
        elif report["status"] == "skipped":
            skipped.append(report)
        else:
            errors.append(report)

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_input_files": len(pnml_files),
        "n_converted": len(converted),
        "n_skipped": len(skipped),
        "n_errors": len(errors),
        "converted": converted,
        "skipped": skipped,
        "errors": errors,
    }


def rename_ptml_files_to_canonical(
    ptml_dir: str | Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Rename PTML files to canonical names:

      lpm.1.3902696586630035445.ptml -> lpm.1.ptml
      lpm.100.307971798554248492.ptml -> lpm.100.ptml

    If canonical target already exists and is different from the source,
    the item is reported as a conflict and not overwritten.
    """
    ptml_dir = Path(ptml_dir)

    if not ptml_dir.exists():
        raise FileNotFoundError(f"PTML directory does not exist: {ptml_dir}")

    ptml_files = sorted(ptml_dir.glob("*.ptml"))

    renamed = []
    skipped = []
    conflicts = []

    for src in ptml_files:
        lpm_id = extract_lpm_id_from_filename(src)

        if lpm_id is None:
            skipped.append(
                {
                    "path": str(src),
                    "reason": "cannot_parse_lpm_id",
                }
            )
            continue

        dst = ptml_dir / f"lpm.{lpm_id}.ptml"

        if src.resolve() == dst.resolve():
            skipped.append(
                {
                    "path": str(src),
                    "reason": "already_canonical",
                }
            )
            continue

        if dst.exists():
            conflicts.append(
                {
                    "source": str(src),
                    "target": str(dst),
                    "reason": "target_exists",
                }
            )
            continue

        if not dry_run:
            shutil.move(str(src), str(dst))

        renamed.append(
            {
                "source": str(src),
                "target": str(dst),
            }
        )

    return {
        "ptml_dir": str(ptml_dir),
        "dry_run": bool(dry_run),
        "n_files": len(ptml_files),
        "n_renamed": len(renamed),
        "n_conflicts": len(conflicts),
        "n_skipped": len(skipped),
        "renamed": renamed,
        "conflicts": conflicts,
        "skipped": skipped,
    }