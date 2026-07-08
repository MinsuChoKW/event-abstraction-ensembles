# src/eae/session/__init__.py

from eae.session.split import (
    build_sessions_from_log,
    load_event_log_dataframe,
    normalize_event_dataframe,
    split_sessions_by_time_gap,
)

from eae.session.encode import (
    attach_session_sequences,
    build_activity_vocabulary,
    encode_sessions_bow,
)

from eae.session.cluster import (
    attach_cluster_labels,
    build_cluster_pattern_table,
    cluster_sessions_dbscan,
    summarize_clusters,
)

from eae.session.export_cuts import (
    build_dataset_cut_dir,
    build_session_output_paths,
    build_topk_pattern_payload,
    export_topk_session_pattern_files,
    load_json_gz,
    save_json,
    save_json_gz,
)