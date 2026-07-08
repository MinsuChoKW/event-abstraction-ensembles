# src/eae/lpm/groups.py

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def validate_lpm_groups(
    group_ids: Sequence[Sequence[int]],
    *,
    allow_empty_group: bool = False,
) -> List[List[int]]:
    """
    Validate and normalize user-defined LPM groups from config.

    Parameters
    ----------
    group_ids
        User-defined LPM group ids loaded from YAML.
        Example:
            [[1, 46], [2, 53], [3, 42]]

    allow_empty_group
        If False, empty groups are not allowed.

    Returns
    -------
    normalized
        List[List[int]] with all ids converted to int.

    Raises
    ------
    ValueError
        If group_ids is invalid.
    """
    if group_ids is None:
        raise ValueError(
            "model_based.lpm_groups is missing. "
            "Please define LPM groups in the config file."
        )

    if not isinstance(group_ids, Sequence) or isinstance(group_ids, (str, bytes)):
        raise ValueError("model_based.lpm_groups must be a list of lists.")

    normalized: List[List[int]] = []

    for gi, group in enumerate(group_ids, start=1):
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            raise ValueError(
                f"LPM group {gi} must be a list of integer ids, got: {type(group)}"
            )

        if len(group) == 0 and not allow_empty_group:
            raise ValueError(f"LPM group {gi} is empty.")

        new_group = []

        for x in group:
            try:
                lpm_id = int(x)
            except Exception as e:
                raise ValueError(
                    f"Invalid LPM id in group {gi}: {x!r}. "
                    "All LPM ids must be integers."
                ) from e

            if lpm_id <= 0:
                raise ValueError(
                    f"Invalid LPM id in group {gi}: {lpm_id}. "
                    "LPM ids must be positive integers."
                )

            new_group.append(lpm_id)

        normalized.append(new_group)

    return normalized


def get_lpm_groups_from_config(cfg: dict) -> List[List[int]]:
    """
    Load user-defined LPM groups from config.

    Expected config path
    --------------------
    cfg["model_based"]["lpm_groups"]
    """
    model_cfg = cfg.get("model_based", {})
    group_ids = model_cfg.get("lpm_groups", None)

    return validate_lpm_groups(group_ids)


def analyze_lpm_group_overlaps(
    group_ids: Sequence[Sequence[int]],
) -> Dict[str, object]:
    """
    Analyze duplicated LPM ids across groups.

    Returns
    -------
    dict
        {
          "n_unique_lpms": int,
          "duplicated_lpms": dict[int, list[int]],
          "pairwise_intersections": list[tuple[int, int, list[int]]]
        }
    """
    group_ids = validate_lpm_groups(group_ids)

    where: Dict[int, List[int]] = defaultdict(list)

    for gi, group in enumerate(group_ids, start=1):
        for lpm_id in group:
            where[int(lpm_id)].append(gi)

    duplicated = {
        lpm_id: groups
        for lpm_id, groups in where.items()
        if len(groups) >= 2
    }

    sets = [set(map(int, group)) for group in group_ids]
    pairwise = []

    for (i, s1), (j, s2) in combinations(list(enumerate(sets, start=1)), 2):
        inter = sorted(s1 & s2)
        if inter:
            pairwise.append((i, j, inter))

    return {
        "n_groups": len(group_ids),
        "n_unique_lpms": len(where),
        "duplicated_lpms": dict(sorted(duplicated.items())),
        "pairwise_intersections": pairwise,
    }


def deduplicate_lpm_group_ids(
    group_ids: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], List[Tuple[int, int, int | None]]]:
    """
    Deduplicate LPM ids by group rank.

    Rule
    ----
    If the same LPM id appears in multiple groups,
    the earlier group keeps it and later groups lose it.

    Returns
    -------
    dedup_group_ids
        Group ids after earlier-group-wins deduplication.

    removed_report
        List of tuples:
        (from_group, lpm_id, kept_group)
    """
    group_ids = validate_lpm_groups(group_ids)

    seen = set()
    dedup_group_ids: List[List[int]] = []
    removed_report: List[Tuple[int, int, int | None]] = []

    for gi, ids in enumerate(group_ids, start=1):
        new_ids = []

        for x in ids:
            x = int(x)

            if x in seen:
                kept_group = None

                for gj, prev in enumerate(dedup_group_ids, start=1):
                    if x in prev:
                        kept_group = gj
                        break

                removed_report.append((gi, x, kept_group))
                continue

            new_ids.append(x)
            seen.add(x)

        dedup_group_ids.append(new_ids)

    return dedup_group_ids, removed_report


def collect_group_ptml_paths(
    ptml_dir: str | Path,
    group_ids: Sequence[Sequence[int]],
    *,
    strict: bool = False,
) -> Tuple[List[List[Path]], List[Tuple[int, int, str]]]:
    """
    Build group -> existing PTML paths.

    Parameters
    ----------
    ptml_dir
        Directory containing files like lpm.1.ptml.

    group_ids
        List of LPM ids per group. Usually loaded from config.

    strict
        If True, raise FileNotFoundError when any expected PTML is missing.
        If False, skip missing files and return a missing report.

    Returns
    -------
    groups_paths
        List of groups, each containing existing PTML Path objects.

    missing_report
        List of tuples:
        (group_index, lpm_id, expected_path)
    """
    group_ids = validate_lpm_groups(group_ids)
    ptml_dir = Path(ptml_dir)

    groups_paths: List[List[Path]] = []
    missing_report: List[Tuple[int, int, str]] = []

    for gi, ids in enumerate(group_ids, start=1):
        paths = []

        for lpm_id in ids:
            p = ptml_dir / f"lpm.{int(lpm_id)}.ptml"

            if p.exists():
                paths.append(p)
            else:
                missing_report.append((gi, int(lpm_id), str(p)))

        groups_paths.append(paths)

    if strict and missing_report:
        first = missing_report[0]
        raise FileNotFoundError(
            f"Missing PTML files. First missing: "
            f"group={first[0]}, lpm={first[1]}, path={first[2]}"
        )

    return groups_paths, missing_report


def summarize_group_paths(
    groups_paths: Sequence[Sequence[Path]],
) -> Dict[str, object]:
    """
    Summarize how many PTML files are kept per group.
    """
    counts = [len(g) for g in groups_paths]

    return {
        "n_groups": len(groups_paths),
        "per_group_counts": counts,
        "n_total_ptml": sum(counts),
        "empty_groups": [i + 1 for i, c in enumerate(counts) if c == 0],
    }