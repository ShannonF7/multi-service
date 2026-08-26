# Multimodal Experiment Plan

This directory now separates the production-oriented multimodal experiment
framework from the older `attractions_db` proof-of-concept scripts.

## Experiment Stages

- E0 `legacy_simclr_128`
  - Existing `src.cv.feature_extractor` SimCLR ResNet50 model.
  - Image vector dimension: 128.
  - Tasks: image-image retrieval, image-node matching through similar historical images.
  - Metrics: Recall@1/5/10, image-node Top-1/Top-5, MRR.

- E1 `openai_clip_vit_b32`
  - Original CLIP cross-modal baseline.
  - Tasks: image-image, text-image, image-text, image-node matching.

- E2 `chinese_clip`
  - Chinese CLIP baseline for Chinese node names, aliases, descriptions and OCR.
  - Compare against E1 to measure Chinese language adaptation.

- E3 `qwen_vl_embedding`
  - Strong multimodal embedding candidate.
  - First implementation should choose one strong model before adding SigLIP 2.

- E4 `fusion_context_v1`
  - Fusion scorer on top of the best embedding model.
  - Features: visual similarity, OCR match, caption-text similarity, node type,
    parent/spatial context and published graph consistency.

## Current Production Data Boundary

- A-side owns actual image files through `ImageAsset`.
- A-side node images are connected through `ImageBinding`.
- B-side structured RAG database already has:
  - `node_assets`
  - `image_embeddings`
  - `clip_image_embeddings`
- This package should use those B-side tables, not the older `attractions_db`
  tables, for production experiments.

## First Commands

Audit synced image assets:

```bash
PYTHONPATH=. python -m src.multimodal.asset_audit --source-scenic-id 4
```

List registered stages:

```python
from src.multimodal.model_registry import list_model_specs
print(list(list_model_specs()))
```


Backfill E0 SimCLR embeddings in dry-run mode:

`ash
PYTHONPATH=. python -m src.multimodal.simclr_backfill --source-scenic-id 4 --limit 20
`

Write E0 SimCLR embeddings:

`ash
PYTHONPATH=. python -m src.multimodal.simclr_backfill --source-scenic-id 4 --limit 20 --write
`

Search by an already embedded asset:

`ash
PYTHONPATH=. python -m src.multimodal.simclr_search --source-scenic-id 4 --asset-id 1158 --top-k 10
`

## E0 Formal Evaluation Loop

Audit image asset distribution:

`ash
PYTHONPATH=. python -m src.multimodal.asset_audit \
  --source-scenic-id 4 \
  --output data/multimodal_eval/scenic_4_asset_audit.json
`

Run duplicate detection smoke test:

`ash
PYTHONPATH=. python -m src.multimodal.image_dedup \
  --source-scenic-id 4 \
  --limit 5 \
  --output data/multimodal_eval/scenic_4_duplicates_smoke.jsonl
`

Build and validate the E0 pilot dataset:

`ash
PYTHONPATH=. python -m src.multimodal.build_image_node_eval_dataset \
  --source-scenic-id 4 \
  --dataset-version scenic_4_e0_pilot_v1 \
  --min-images-per-node 3 \
  --query-ratio 0.3 \
  --seed 20260730 \
  --max-nodes 30 \
  --output data/multimodal_eval/scenic_4_e0_pilot_v1.jsonl

PYTHONPATH=. python -m src.multimodal.validate_image_node_eval_dataset \
  --dataset data/multimodal_eval/scenic_4_e0_pilot_v1.jsonl
`

Backfill only the pilot dataset and evaluate E0:

`ash
PYTHONPATH=. python -m src.multimodal.simclr_backfill \
  --dataset data/multimodal_eval/scenic_4_e0_pilot_v1.jsonl \
  --write

PYTHONPATH=. python -m src.multimodal.evaluate_simclr_e0 \
  --dataset data/multimodal_eval/scenic_4_e0_pilot_v1.jsonl \
  --aggregation max,top3_mean \
  --top-k 1,3,5,10 \
  --output-dir outputs/multimodal/e0_simclr_pilot
`

## E0-A Cross-Domain Update

The current SimCLR model is recorded as simclr_source_scenic_v1_128:

`	ext
stage = E0-A
training_domain = source_scenic
evaluation_domain = scenic_4
target_domain_seen = false
evaluation_setting = cross_domain_zero_shot
`

Pilot v1 is frozen. Pilot v2 adds isual_label_status and uses closed-set node ranking:

`ash
PYTHONPATH=. python -m src.multimodal.build_image_node_eval_dataset \
  --source-scenic-id 4 \
  --dataset-version scenic_4_e0_pilot_v2 \
  --min-images-per-node 3 \
  --query-ratio 0.3 \
  --seed 20260730 \
  --max-nodes 30 \
  --output data/multimodal_eval/scenic_4_e0_pilot_v2.jsonl \
  --default-visual-label-status direct

PYTHONPATH=. python -m src.multimodal.audit_cross_node_duplicates \
  --dataset data/multimodal_eval/scenic_4_e0_pilot_v2.jsonl \
  --output-dir outputs/multimodal/e0_a_cross_domain_pilot_v2/duplicate_audit

PYTHONPATH=. python -m src.multimodal.evaluate_simclr_e0 \
  --dataset data/multimodal_eval/scenic_4_e0_pilot_v2.jsonl \
  --model simclr_source_scenic_v1_128 \
  --evaluation-mode closed-set \
  --aggregations max,centroid,top3_mean,hybrid \
  --top-k 1,3,5,10 \
  --output-dir outputs/multimodal/e0_a_cross_domain_pilot_v2
`
