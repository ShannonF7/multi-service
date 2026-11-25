"""Append verified resource tables to the restructured comparison report."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs/multimodal/scenic_4_image_node_expanded_v3")
report = ROOT / "EXPANDED_V3_FINAL_TABLES.md"
data = json.loads((ROOT / "expanded_v3_final_tables.json").read_text(encoding="utf-8"))
text = report.read_text(encoding="utf-8")
marker = "## Table D: Fixed Batch Resource Benchmark"
if marker in text:
    text = text.split(marker, 1)[0].rstrip() + "\n"


def gb(value):
    return float(value) / 1024 / 1024 / 1024


def append_table(lines, title, rows):
    lines.extend([
        "", title, "",
        "| Model | Disk GB | Peak VRAM MB | Image batch | Image sec/395 | Image/s | Text batch | Text sec/440 | Text/s | Dim | Precision | GPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['model']} | {gb(row['model_disk_size_bytes']):.3f} | {row['peak_memory_mb']:.1f} | "
            f"{row['image_batch_size']} | {row['image_encode_seconds']:.3f} | {row['image_throughput_per_second']:.3f} | "
            f"{row['text_batch_size']} | {row['text_encode_seconds']:.3f} | {row['text_throughput_per_second']:.3f} | "
            f"{row['image_vector_dim']} | {row['precision']} | {row['gpu_name']} |"
        )


lines = []
append_table(lines, "## Table D: Fixed Batch Resource Benchmark (Image=8, Text=1)", data["table_d_fixed_batch"])
append_table(lines, "## Table E: Stable Maximum Throughput", data["table_e_max_throughput"])
lines.extend([
    "",
    "Qwen's current stable maximum batch equals the fixed protocol (image=8, text=1).",
    "Qwen image and text encoding use isolated model instances because its native processor is stateful across modalities.",
])
report.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"report": str(report), "resource_tables": ["D", "E"]}))
