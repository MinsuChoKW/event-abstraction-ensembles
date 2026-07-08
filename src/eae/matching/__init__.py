# src/eae/matching/__init__.py

from eae.matching.alignment import alignment_cost

from eae.matching.segment_match import (
    build_edges_forward_for_case,
    flatten_pattern_pool,
    index_patterns_by_length,
)

from eae.matching.bidirectional import (
    build_edges_backward_for_case,
    merge_forward_backward_edges,
    remap_reversed_edge_to_original,
    reverse_pattern_pool,
)

from eae.matching.dp_runner import (
    build_segment_matches_for_case,
    build_segment_matches_for_cases,
    write_jsonl_gz,
)