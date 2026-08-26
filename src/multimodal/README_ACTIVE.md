# Active Multimodal Source

This directory contains the reproducible expanded_v3 multimodal pipeline.

Key entrypoints:
- `benchmark_multimodal_resources_unified.py`: unified resource benchmark.
- `evaluate_*`: E0-E4 accuracy evaluations.
- `build_node_text_profiles.py` / `_en.py`: T1-T4 profiles.
- `freeze_expanded_v3_images.py`: persist final raw images and checksums.
- `restructure_comparison_tables.py`: generate A-1 to C-2 comparison tables.
- `append_resource_tables.py`: append resource Tables D/E.

Legacy Graphify, generic test data, backups and bytecode are in the external cleanup archive.
