# src/eae/discovery/expansion.py

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Tuple

from pm4py.objects.petri_net.obj import (
    Marking,
    PetriNet,
)
from pm4py.objects.petri_net.utils import (
    petri_utils,
)

from eae.discovery.model_discovery_algo import (
    canonicalize_petri_net,
    save_petri_net,
)


def _place_key(
    place: PetriNet.Place,
) -> str:
    return str(
        getattr(place, "name", "")
    )


def _transition_key(
    transition: PetriNet.Transition,
):
    return (
        str(transition.label)
        if transition.label is not None
        else "",
        str(transition.name),
    )


def _arc_key(arc):
    return (
        str(
            getattr(
                arc.source,
                "name",
                "",
            )
        ),
        str(
            getattr(
                arc.target,
                "name",
                "",
            )
        ),
    )


def remap_marking_to_net(
    net: PetriNet,
    marking: Marking,
) -> Marking:
    """
    Reconnect a marking to the place objects of a copied net.
    """
    place_by_name = {
        place.name: place
        for place in net.places
    }

    new_marking = Marking()

    for old_place, count in marking.items():
        if old_place.name not in place_by_name:
            raise KeyError(
                f"Place {old_place.name} "
                "from marking not found in net."
            )

        new_marking[
            place_by_name[old_place.name]
        ] = count

    return new_marking


def replace_transition_with_subnet(
    *,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    transition: PetriNet.Transition,
    subnet: PetriNet,
    subnet_initial_marking: Marking,
    subnet_final_marking: Marking,
    suffix: str,
) -> Tuple[PetriNet, Marking, Marking]:
    """
    Original notebook EXP insertion.

    The old high-level transition is replaced by:

      predecessor places
        → silent start transition
        → copied local subnet
        → silent end transition
        → successor places
    """
    incoming_places = sorted(
        [
            arc.source
            for arc in list(
                transition.in_arcs
            )
        ],
        key=_place_key,
    )

    outgoing_places = sorted(
        [
            arc.target
            for arc in list(
                transition.out_arcs
            )
        ],
        key=_place_key,
    )

    place_map: Dict[
        PetriNet.Place,
        PetriNet.Place,
    ] = {}

    transition_map: Dict[
        PetriNet.Transition,
        PetriNet.Transition,
    ] = {}

    for place in sorted(
        list(subnet.places),
        key=_place_key,
    ):
        new_place = PetriNet.Place(
            f"{place.name}__{suffix}"
        )

        net.places.add(new_place)
        place_map[place] = new_place

    for local_transition in sorted(
        list(subnet.transitions),
        key=_transition_key,
    ):
        new_transition = (
            PetriNet.Transition(
                (
                    f"{local_transition.name}"
                    f"__{suffix}"
                ),
                local_transition.label,
            )
        )

        net.transitions.add(
            new_transition
        )
        transition_map[
            local_transition
        ] = new_transition

    for arc in sorted(
        list(subnet.arcs),
        key=_arc_key,
    ):
        source = (
            place_map[arc.source]
            if isinstance(
                arc.source,
                PetriNet.Place,
            )
            else transition_map[
                arc.source
            ]
        )

        target = (
            place_map[arc.target]
            if isinstance(
                arc.target,
                PetriNet.Place,
            )
            else transition_map[
                arc.target
            ]
        )

        petri_utils.add_arc_from_to(
            source,
            target,
            net,
        )

    subnet_start_places = sorted(
        [
            place_map[place]
            for place in (
                subnet_initial_marking.keys()
            )
        ],
        key=_place_key,
    )

    subnet_end_places = sorted(
        [
            place_map[place]
            for place in (
                subnet_final_marking.keys()
            )
        ],
        key=_place_key,
    )

    tau_start = PetriNet.Transition(
        f"{suffix}__TAU_START",
        None,
    )

    tau_end = PetriNet.Transition(
        f"{suffix}__TAU_END",
        None,
    )

    net.transitions.add(tau_start)
    net.transitions.add(tau_end)

    for place in incoming_places:
        petri_utils.add_arc_from_to(
            place,
            tau_start,
            net,
        )

    for place in subnet_start_places:
        petri_utils.add_arc_from_to(
            tau_start,
            place,
            net,
        )

    for place in subnet_end_places:
        petri_utils.add_arc_from_to(
            place,
            tau_end,
            net,
        )

    for place in outgoing_places:
        petri_utils.add_arc_from_to(
            tau_end,
            place,
            net,
        )

    petri_utils.remove_transition(
        net,
        transition,
    )

    return (
        net,
        initial_marking,
        final_marking,
    )


def expand_abstract_petri_net(
    net_abstract: PetriNet,
    initial_marking_abstract: Marking,
    final_marking_abstract: Marking,
    local_models_by_label: Dict[
        str,
        Tuple[
            PetriNet,
            Marking,
            Marking,
        ],
    ],
):
    """
    Replace every visible ABS transition whose label has a local model.
    """
    (
        net,
        initial_marking,
        final_marking,
    ) = copy.deepcopy(
        (
            net_abstract,
            initial_marking_abstract,
            final_marking_abstract,
        )
    )

    initial_marking = remap_marking_to_net(
        net,
        initial_marking,
    )

    final_marking = remap_marking_to_net(
        net,
        final_marking,
    )

    replaced_labels = []
    missing_labels = []
    replaced_occurrences = []

    transitions_to_check = sorted(
        [
            transition
            for transition in list(
                net.transitions
            )
            if transition.label is not None
        ],
        key=_transition_key,
    )

    occurrence_counter: Dict[
        str,
        int,
    ] = {}

    for old_transition in transitions_to_check:
        label = str(
            old_transition.label
        )

        if label not in local_models_by_label:
            missing_labels.append(label)
            continue

        (
            local_net,
            local_initial_marking,
            local_final_marking,
        ) = local_models_by_label[label]

        occurrence_counter[label] = (
            occurrence_counter.get(
                label,
                0,
            )
            + 1
        )

        occurrence = occurrence_counter[
            label
        ]

        suffix = (
            f"EXPINS_{label}"
            f"__OCC{occurrence:03d}"
        )

        (
            net,
            initial_marking,
            final_marking,
        ) = replace_transition_with_subnet(
            net=net,
            initial_marking=(
                initial_marking
            ),
            final_marking=(
                final_marking
            ),
            transition=old_transition,
            subnet=local_net,
            subnet_initial_marking=(
                local_initial_marking
            ),
            subnet_final_marking=(
                local_final_marking
            ),
            suffix=suffix,
        )

        replaced_labels.append(label)

        replaced_occurrences.append(
            {
                "label": label,
                "occurrence": occurrence,
                "suffix": suffix,
            }
        )

    net, initial_marking, final_marking = (
        canonicalize_petri_net(
            net,
            initial_marking,
            final_marking,
        )
    )

    metadata = {
        "replaced_labels": sorted(
            set(replaced_labels)
        ),
        "missing_labels": sorted(
            set(missing_labels)
        ),
        "n_replaced": int(
            len(replaced_labels)
        ),
        "n_missing": int(
            len(set(missing_labels))
        ),
        "replaced_occurrences": (
            replaced_occurrences
        ),
        "places": int(len(net.places)),
        "transitions": int(
            len(net.transitions)
        ),
        "arcs": int(len(net.arcs)),
    }

    return (
        net,
        initial_marking,
        final_marking,
        metadata,
    )


def save_expanded_model(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    *,
    pnml_path: str | Path,
    meta_path: str | Path,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    import json

    net_meta = save_petri_net(
        net,
        initial_marking,
        final_marking,
        pnml_path=pnml_path,
    )

    meta_path = Path(meta_path)
    meta_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_metadata = {
        **metadata,
        **net_meta,
    }

    meta_path.write_text(
        json.dumps(
            final_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return final_metadata