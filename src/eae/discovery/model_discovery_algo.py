# src/eae/discovery/model_discovery_algo.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pm4py

from pm4py.objects.petri_net.obj import (
    Marking,
    PetriNet,
)
from pm4py.objects.petri_net.utils import petri_utils


def load_petri_net(
    pnml_path: str | Path,
):
    pnml_path = Path(pnml_path)

    if not pnml_path.exists():
        raise FileNotFoundError(
            f"PNML file does not exist: {pnml_path}"
        )

    try:
        return pm4py.read_pnml(
            str(pnml_path)
        )
    except Exception:
        from pm4py.objects.petri_net.importer import (
            importer as pnml_importer,
        )

        return pnml_importer.apply(
            str(pnml_path)
        )

def discover_process_tree_and_petri(
    log,
    *,
    noise_threshold: float = 0.0,
):
    """
    Original notebook Inductive Miner wrapper.

    Process tree is discovered first and then converted to a Petri net.
    """
    from pm4py.algo.discovery.inductive import (
        algorithm as inductive_miner,
    )
    from pm4py.objects.conversion.process_tree import (
        converter as process_tree_converter,
    )

    try:
        tree = pm4py.discover_process_tree_inductive(
            log,
            noise_threshold=float(noise_threshold),
            multi_processing=False,
        )

    except TypeError:
        try:
            tree = pm4py.discover_process_tree_inductive(
                log,
                noise_threshold=float(noise_threshold),
            )
        except TypeError:
            tree = pm4py.discover_process_tree_inductive(
                log
            )

    except Exception:
        try:
            tree = inductive_miner.apply(
                log,
                parameters={
                    "noise_threshold": float(
                        noise_threshold
                    ),
                    "multi_processing": False,
                },
            )
        except Exception:
            tree = inductive_miner.apply(
                log,
                parameters={
                    "noise_threshold": float(
                        noise_threshold
                    )
                },
            )

    net, initial_marking, final_marking = (
        process_tree_converter.apply(
            tree,
            variant=(
                process_tree_converter
                .Variants
                .TO_PETRI_NET
            ),
        )
    )

    return (
        tree,
        net,
        initial_marking,
        final_marking,
    )


def canonicalize_petri_net(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> Tuple[PetriNet, Marking, Marking]:
    """
    Deterministically rebuild a Petri net.

    This follows the notebook's canonicalization step used before
    inserting local models.
    """
    new_net = PetriNet(
        net.name or "canonical"
    )

    place_map: Dict[Any, PetriNet.Place] = {}

    for place in sorted(
        net.places,
        key=lambda x: str(x.name),
    ):
        new_place = PetriNet.Place(
            str(place.name)
        )
        new_net.places.add(new_place)
        place_map[place] = new_place

    transition_map: Dict[
        Any,
        PetriNet.Transition,
    ] = {}

    for transition in sorted(
        net.transitions,
        key=lambda x: (
            str(x.label)
            if x.label is not None
            else "",
            str(x.name),
        ),
    ):
        new_transition = PetriNet.Transition(
            str(transition.name),
            transition.label,
        )
        new_net.transitions.add(new_transition)
        transition_map[transition] = new_transition

    for arc in sorted(
        net.arcs,
        key=lambda a: (
            str(
                getattr(
                    a.source,
                    "name",
                    "",
                )
            ),
            str(
                getattr(
                    a.target,
                    "name",
                    "",
                )
            ),
        ),
    ):
        source = (
            place_map[arc.source]
            if isinstance(
                arc.source,
                PetriNet.Place,
            )
            else transition_map[arc.source]
        )

        target = (
            place_map[arc.target]
            if isinstance(
                arc.target,
                PetriNet.Place,
            )
            else transition_map[arc.target]
        )

        petri_utils.add_arc_from_to(
            source,
            target,
            new_net,
        )

    new_initial_marking = Marking()

    for place, count in sorted(
        initial_marking.items(),
        key=lambda item: str(item[0].name),
    ):
        new_initial_marking[
            place_map[place]
        ] = count

    new_final_marking = Marking()

    for place, count in sorted(
        final_marking.items(),
        key=lambda item: str(item[0].name),
    ):
        new_final_marking[
            place_map[place]
        ] = count

    return (
        new_net,
        new_initial_marking,
        new_final_marking,
    )


def discover_stable_model(
    log,
    *,
    noise_threshold: float = 0.0,
):
    tree, net, initial_marking, final_marking = (
        discover_process_tree_and_petri(
            log,
            noise_threshold=noise_threshold,
        )
    )

    net, initial_marking, final_marking = (
        canonicalize_petri_net(
            net,
            initial_marking,
            final_marking,
        )
    )

    return (
        tree,
        net,
        initial_marking,
        final_marking,
    )


def save_process_tree(
    tree,
    *,
    text_path: str | Path,
    ptml_path: str | Path,
) -> Dict[str, Any]:
    text_path = Path(text_path)
    ptml_path = Path(ptml_path)

    text_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ptml_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text_path.write_text(
        str(tree),
        encoding="utf-8",
    )

    ptml_error = None

    try:
        from pm4py.objects.process_tree.exporter import (
            exporter as process_tree_exporter,
        )

        process_tree_exporter.apply(
            tree,
            str(ptml_path),
        )

    except Exception as exc:
        ptml_error = repr(exc)

        ptml_path.with_suffix(
            ".ptml.ERROR.txt"
        ).write_text(
            ptml_error,
            encoding="utf-8",
        )

    return {
        "tree_text_path": str(text_path),
        "ptml_path": str(ptml_path),
        "ptml_error": ptml_error,
    }


def save_petri_net(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    *,
    pnml_path: str | Path,
    meta_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Save an accepting Petri net as PNML.

    The final marking is preserved across different PM4Py
    exporter signatures whenever possible.
    """
    from pm4py.objects.petri_net.exporter import (
        exporter as pnml_exporter,
    )

    pnml_path = Path(pnml_path)
    pnml_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_error = None

    try:
        # Newer PM4Py versions
        pnml_exporter.apply(
            net,
            initial_marking,
            str(pnml_path),
            final_marking=final_marking,
        )

    except TypeError:
        try:
            # Some versions accept final marking through parameters
            pnml_exporter.apply(
                net,
                initial_marking,
                str(pnml_path),
                parameters={
                    "final_marking": final_marking,
                },
            )

        except TypeError:
            try:
                # High-level PM4Py API fallback
                pm4py.write_pnml(
                    net,
                    initial_marking,
                    final_marking,
                    str(pnml_path),
                )

            except Exception as exc:
                export_error = repr(exc)

                raise RuntimeError(
                    "Failed to export Petri net with "
                    "initial and final markings.\n"
                    f"Output path: {pnml_path}\n"
                    f"Error: {export_error}"
                ) from exc

    if not pnml_path.exists():
        raise RuntimeError(
            f"PNML export did not create a file: {pnml_path}"
        )

    meta = {
        "pnml_path": str(pnml_path),
        "places": int(len(net.places)),
        "transitions": int(
            len(net.transitions)
        ),
        "arcs": int(len(net.arcs)),
        "initial_marking_places": int(
            len(initial_marking)
        ),
        "final_marking_places": int(
            len(final_marking)
        ),
    }

    if meta_path is not None:
        meta_path = Path(meta_path)
        meta_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        meta_path.write_text(
            json.dumps(
                meta,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return meta