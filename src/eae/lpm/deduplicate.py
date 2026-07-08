# src/eae/lpm/deduplicate.py

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


PathTuple = Tuple[str, ...]


def path_key(path_list: Sequence[str]) -> PathTuple:
    """
    Convert list path to hashable tuple.
    """
    return tuple(str(x) for x in path_list)


def build_group_lpm_path_sets(
    payload: Dict[str, object],
) -> Dict[str, Dict[str, Set[PathTuple]]]:
    """
    Convert payload groups to:
        group -> lpm_id -> set(path_tuple)
    """
    groups = payload.get("groups", {})
    out: Dict[str, Dict[str, Set[PathTuple]]] = {}

    for gkey in sorted(groups.keys()):
        out[gkey] = {}

        for lpm_id, item in groups[gkey].items():
            paths = item.get("paths", [])
            out[gkey][str(lpm_id)] = {path_key(p) for p in paths}

    return out


def compute_path_overlap_stats(
    group_lpm_paths: Dict[str, Dict[str, Set[PathTuple]]],
) -> Dict[str, object]:
    """
    Compute global unique paths, per-group counts,
    and pairwise overlaps between groups.
    """
    gkeys = sorted(group_lpm_paths.keys())

    group_sets = {}

    for g in gkeys:
        s: Set[PathTuple] = set()

        for _, paths in group_lpm_paths[g].items():
            s |= paths

        group_sets[g] = s

    all_union: Set[PathTuple] = set()
    for g in gkeys:
        all_union |= group_sets[g]

    per_group_unique = [(g, len(group_sets[g])) for g in gkeys]
    per_group_total = [
        (g, sum(len(paths) for paths in group_lpm_paths[g].values()))
        for g in gkeys
    ]

    pair_overlaps = []

    for g1, g2 in combinations(gkeys, 2):
        inter = group_sets[g1] & group_sets[g2]

        if inter:
            pair_overlaps.append((g1, g2, len(inter)))

    pair_overlaps.sort(key=lambda x: -x[2])

    path_to_groups: Dict[PathTuple, Set[str]] = defaultdict(set)

    for g in gkeys:
        for p in group_sets[g]:
            path_to_groups[p].add(g)

    shared_paths = [
        {
            "path": list(p),
            "n_groups": len(groups),
            "groups": sorted(groups),
        }
        for p, groups in path_to_groups.items()
        if len(groups) >= 2
    ]

    shared_paths.sort(
        key=lambda row: (
            -row["n_groups"],
            len(row["path"]),
            row["path"],
        )
    )

    return {
        "global_unique_paths": len(all_union),
        "per_group_unique": per_group_unique,
        "per_group_total": per_group_total,
        "pair_overlaps": pair_overlaps,
        "shared_paths": shared_paths,
    }


def deduplicate_paths_by_rank(
    group_lpm_paths: Dict[str, Dict[str, Set[PathTuple]]],
) -> Tuple[
    Dict[str, Dict[str, Set[PathTuple]]],
    Dict[str, object],
]:
    """
    Deduplicate paths by:
        1. earlier group wins
        2. within the same group, earlier LPM id wins

    This corresponds to the original notebook's final path dedup logic.
    """
    gkeys = sorted(group_lpm_paths.keys())

    seen_global: Set[PathTuple] = set()
    dedup_map: Dict[str, Dict[str, Set[PathTuple]]] = {}

    removed_report = []
    kept_report = []

    for g in gkeys:
        dedup_map[g] = {}

        lpm_sorted = sorted(group_lpm_paths[g].keys(), key=lambda x: int(x))

        for lpm_id in lpm_sorted:
            paths = group_lpm_paths[g][lpm_id]
            kept: Set[PathTuple] = set()

            for p in paths:
                if p in seen_global:
                    continue

                kept.add(p)
                seen_global.add(p)

            dedup_map[g][lpm_id] = kept

            removed_report.append(
                {
                    "group": g,
                    "lpm_id": int(lpm_id),
                    "removed": len(paths) - len(kept),
                }
            )

            kept_report.append(
                {
                    "group": g,
                    "lpm_id": int(lpm_id),
                    "kept": len(kept),
                }
            )

    stats_after = compute_path_overlap_stats(dedup_map)

    internal_duplicates = {}

    for g in gkeys:
        total = sum(len(ps) for ps in dedup_map[g].values())
        union: Set[PathTuple] = set()

        for ps in dedup_map[g].values():
            union |= ps

        internal_duplicates[g] = total - len(union)

    report = {
        "dedup_rule": "earlier_group_wins, within_group earlier_lpm_wins",
        "removed_report": removed_report,
        "kept_report": kept_report,
        "stats_after": stats_after,
        "internal_duplicates": internal_duplicates,
    }

    return dedup_map, report


def build_dedup_payload(
    original_payload: Dict[str, object],
    dedup_map: Dict[str, Dict[str, Set[PathTuple]]],
) -> Dict[str, object]:
    """
    Build final JSON payload with same structure as the raw payload,
    but with deduplicated paths.
    """
    groups_src = original_payload.get("groups", {})
    gkeys = sorted(dedup_map.keys())

    meta = dict(original_payload.get("meta", {}))
    meta.update(
        {
            "created_at": datetime.now().isoformat(),
            "dedup_rule": "earlier_group_wins, within_group earlier_lpm_wins",
            "dedup_level": "path_unique_across_groups_and_lpms",
        }
    )

    groups_out = {}
    sum_paths = 0
    global_union: Set[PathTuple] = set()

    for g in gkeys:
        groups_out[g] = {}

        lpm_ids_sorted = sorted(dedup_map[g].keys(), key=lambda x: int(x))

        for lpm_id in lpm_ids_sorted:
            kept_set = dedup_map[g][lpm_id]

            kept_paths = [
                list(t)
                for t in sorted(kept_set, key=lambda x: (len(x), x))
            ]

            src_item = groups_src[g][lpm_id]

            groups_out[g][lpm_id] = {
                "lpm_id": int(src_item.get("lpm_id", int(lpm_id))),
                "ptml_path": src_item.get("ptml_path", None),
                "n_paths": int(len(kept_paths)),
                "paths": kept_paths,
            }

            sum_paths += len(kept_paths)
            global_union |= kept_set

    payload_final = {
        "meta": meta,
        "groups": groups_out,
    }

    payload_final["meta"].update(
        {
            "global_unique_paths": len(global_union),
            "sum_paths_over_lpms": sum_paths,
            "sanity_equal_global_unique_and_sum": len(global_union) == sum_paths,
        }
    )

    return payload_final


def save_json(
    obj: Dict[str, object],
    output_path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """
    Save object as JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

    return output_path


def load_json(path: str | Path) -> Dict[str, object]:
    """
    Load JSON file.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deduplicate_lpm_payload_file(
    input_json: str | Path,
    output_json: str | Path,
) -> Dict[str, object]:
    """
    One-shot utility:
        raw lpm_group_paths_loop{n}.json
        -> deduplicated lpm_group_paths_loop{n}_remove_dup.json
    """
    payload = load_json(input_json)

    orig_map = build_group_lpm_path_sets(payload)
    stats_before = compute_path_overlap_stats(orig_map)

    dedup_map, dedup_report = deduplicate_paths_by_rank(orig_map)
    final_payload = build_dedup_payload(payload, dedup_map)

    save_json(final_payload, output_json)

    return {
        "input_json": str(input_json),
        "output_json": str(output_json),
        "stats_before": stats_before,
        "dedup_report": dedup_report,
        "final_meta": final_payload.get("meta", {}),
    }