"""Separate controlled model comparisons from optimized system comparisons."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


OUT = Path("outputs/multimodal")
ROOT = OUT / "scenic_4_image_node_expanded_v3"

PURE = {
    "SimCLR": OUT / "e0_simclr_fresh_expanded_v3_frozen_test_strict/metrics.json",
    "OpenAI CLIP": OUT / "e1_openai_clip_vit_base_patch16_expanded_v3_frozen_test/metrics.json",
    "Chinese-CLIP": OUT / "e2_chinese_clip_vit_base_patch16_expanded_v3_frozen_test/metrics.json",
    "SigLIP2": OUT / "e3_siglip2_base_patch16_224_expanded_v3_frozen_test/metrics.json",
    "Qwen3-VL-Embedding-2B": OUT / "e4_qwen3_vl_embedding_2b_expanded_v3_frozen_test/metrics.json",
}
CONTROLLED = {
    "OpenAI CLIP": OUT / "e1_openai_clip_vit_base_patch16_text_zh_expanded_v3_frozen_test/metrics.json",
    "Chinese-CLIP": OUT / "e2_chinese_clip_vit_base_patch16_text_zh_expanded_v3_frozen_test/metrics.json",
    "SigLIP2": OUT / "e3_siglip2_base_patch16_224_text_zh_expanded_v3_frozen_test/metrics.json",
    "Qwen3-VL-Embedding-2B": OUT / "e4_qwen3_vl_embedding_2b_text_zh_expanded_v3_frozen_test/metrics.json",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_row(model, metrics):
    value = metrics["closed_set"]["image_retrieval"]
    return {"model": model, **value}


def node_row(model, strategy, metrics):
    value = metrics["closed_set"]["node_matching"][strategy]
    return {
        "model": model,
        "strategy": strategy,
        "top1": value["strict_top1"],
        "top5": value["strict_top5"],
        "top10": value["strict_top10"],
        "mrr": value["mrr"],
    }


canonical_path = ROOT / "expanded_v3_final_tables.json"
report_path = ROOT / "EXPANDED_V3_FINAL_TABLES.md"
canonical = load(canonical_path)
pure_metrics = {model: load(path) for model, path in PURE.items()}
controlled_metrics = {model: load(path) for model, path in CONTROLLED.items()}

queries = {value["query_count"] for value in controlled_metrics.values()}
galleries = {value["gallery_count"] for value in controlled_metrics.values()}
profiles = {value["profile_count"] for value in controlled_metrics.values()}
text_queries = {value["text_query_node_count"] for value in controlled_metrics.values()}
datasets = {value["dataset"] for value in controlled_metrics.values()}
if (queries, galleries, profiles, text_queries) != ({73}, {246}, {440}, {64}):
    raise RuntimeError("controlled comparison inputs differ")
if len(datasets) != 1:
    raise RuntimeError(f"controlled datasets differ: {datasets}")

table_a1 = [retrieval_row(model, value) for model, value in pure_metrics.items()]
table_a2 = [node_row(model, "hybrid", value) for model, value in pure_metrics.items()]
table_a3 = [
    node_row(model, strategy, value)
    for model, value in pure_metrics.items()
    for strategy in ("max", "top3_mean", "hybrid")
]
table_b1 = []
table_c1 = []
for model, value in controlled_metrics.items():
    variant = value["variants"]["T4"]
    table_b1.append({
        "model": model, "queries": 73, "candidates": 110,
        "language": "zh", "template": "T4", "fusion": "none",
        **variant["image_to_node_text"],
    })
    table_c1.append({
        "model": model, "queries": 64, "candidates": 246,
        "language": "zh", "template": "T4", "fusion": "none",
        **variant["node_text_to_image"],
    })

restructured = {
    **canonical,
    "comparison_scope": {
        "table_a1": "image encoder comparison without node aggregation",
        "table_a2": "image encoder plus one shared hybrid node aggregation strategy",
        "table_a3": "node aggregation ablation on the same Frozen Test",
        "table_b1": "controlled image-to-node-text comparison using Chinese T4 without fusion",
        "table_b2": "optimized system comparison using each model's Validation-selected configuration",
        "table_c1": "controlled node-text-to-image comparison using Chinese T4 without fusion",
        "table_c2": "optimized system comparison using each model's Validation-selected configuration",
        "hybrid_formula": "0.7 * max_score + 0.3 * mean(top3_scores)",
    },
    "table_a1_image_retrieval": table_a1,
    "table_a2_node_hybrid": table_a2,
    "table_a3_aggregation_ablation": table_a3,
    "table_b1_controlled_zh_t4": table_b1,
    "table_b2_best_system": canonical["table_b"],
    "table_c1_controlled_zh_t4": table_c1,
    "table_c2_best_system": canonical["table_c"],
}

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2(canonical_path, canonical_path.with_name(f"{canonical_path.name}.bak_{stamp}"))
shutil.copy2(report_path, report_path.with_name(f"{report_path.name}.bak_{stamp}"))
canonical_path.write_text(json.dumps(restructured, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Expanded v3 Final Comparison Tables", "",
    "## Interpretation", "",
    "- Table A-1 compares image encoders only.",
    "- Table A-2 compares model + the shared Hybrid node aggregation system.",
    "- Hybrid = `0.7 * max_score + 0.3 * mean(top3_scores)`.",
    "- Tables B-1/C-1 are controlled model comparisons: Chinese T4, no fusion, identical candidates.",
    "- Tables B-2/C-2 are best-config system comparisons, not isolated model comparisons.",
    "- Language, template and alpha are selected on Validation only; Frozen evaluates only the selected configuration.",
]


def append_metrics(title, rows, first_columns):
    lines.extend(["", title, ""])
    headers = [item[0] for item in first_columns] + ["R@1", "R@5", "R@10", "MRR"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|")
    for row in rows:
        prefix = [str(row[key]) for _, key in first_columns]
        values = [row["recall@1"], row["recall@5"], row["recall@10"], row["mrr"]]
        lines.append("| " + " | ".join(prefix + [f"{value:.4f}" for value in values]) + " |")


append_metrics("## Table A-1: Image-to-Image Retrieval", table_a1, [("Model", "model")])

lines.extend(["", "## Table A-2: Image-to-Node with Shared Hybrid", "", "| Model | Node Top1 | Top5 | Top10 | MRR |", "|---|---:|---:|---:|---:|"])
for row in table_a2:
    lines.append(f"| {row['model']} | {row['top1']:.4f} | {row['top5']:.4f} | {row['top10']:.4f} | {row['mrr']:.4f} |")

lines.extend(["", "## Table A-3: Node Aggregation Ablation", "", "| Model | Aggregation | Top1 | Top5 | Top10 | MRR |", "|---|---|---:|---:|---:|---:|"])
for row in table_a3:
    lines.append(f"| {row['model']} | {row['strategy']} | {row['top1']:.4f} | {row['top5']:.4f} | {row['top10']:.4f} | {row['mrr']:.4f} |")

append_metrics("## Table B-1: Controlled Image-to-Node-Text (Chinese T4, No Fusion)", table_b1, [("Model", "model"), ("Queries", "queries"), ("Candidates", "candidates")])
append_metrics("## Table B-2: Best Validation-Selected Image-to-Node-Text System", canonical["table_b"], [("Model", "model"), ("Queries", "queries"), ("Candidates", "candidates"), ("Configuration", "language_or_fusion"), ("Template", "template")])
append_metrics("## Table C-1: Controlled Node-Text-to-Image (Chinese T4, No Fusion)", table_c1, [("Model", "model"), ("Queries", "queries"), ("Candidates", "candidates")])
append_metrics("## Table C-2: Best Validation-Selected Node-Text-to-Image System", canonical["table_c"], [("Model", "model"), ("Queries", "queries"), ("Candidates", "candidates"), ("Configuration", "language_or_fusion"), ("Template", "template")])

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({
    "report": str(report_path), "json": str(canonical_path),
    "controlled_dataset": next(iter(datasets)),
    "image_queries": 73, "text_queries": 64, "node_candidates": 110, "gallery_candidates": 246,
}, ensure_ascii=False, indent=2))
