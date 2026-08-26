import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import create_engine, text

# 兼容直接执行：python /abs/path/src/multimodal/run_dataset_eval.py
# 此时 sys.path 通常只包含脚本目录，需要补充项目根目录以导入 src.*
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from src.multimodal.image_retrieval_pipeline import (
    DBConfig,
    extract_embedding,
    insert_attraction_image,
    search_similar_images,
)

DATASET_ROOT = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/pgvector_optimized/src/multimodal/test_dataset"
REPORT_JSON = os.path.join(DATASET_ROOT, "eval_report.json")
REPORT_MD = os.path.join(DATASET_ROOT, "eval_report.md")

BASE_LABELS = ["architecture", "city", "nature", "animal", "vehicle"]
QUERY_SPLITS = ["normal", "hard", "negative"]
TOP_K = 5
UPLOAD_BY = "multimodal_dataset_eval"


@dataclass
class InsertedBaseImage:
    image_id: int
    label: str
    file_path: str
    attraction_id: int


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, f)) and not f.startswith(".")
    ]
    return files


def ensure_eval_attraction(conn, label: str) -> int:
    name = f"[MM_EVAL] {label}"
    row = conn.execute(
        text("SELECT id FROM attractions WHERE name = :name LIMIT 1"),
        {"name": name},
    ).fetchone()
    if row:
        return int(row.id)

    # 满足 not null 字段：name, upload_by
    row = conn.execute(
        text(
            """
            INSERT INTO attractions (name, description, position, jpg_path, upload_by)
            VALUES (:name, :description, :position, :jpg_path, :upload_by)
            RETURNING id
            """
        ),
        {
            "name": name,
            "description": "multimodal eval label",
            "position": label,
            "jpg_path": None,
            "upload_by": UPLOAD_BY,
        },
    ).fetchone()
    return int(row.id)


def seed_base_images(conn) -> Tuple[Dict[str, int], List[InsertedBaseImage]]:
    label_to_attr_id: Dict[str, int] = {}
    inserted: List[InsertedBaseImage] = []

    for label in BASE_LABELS:
        attr_id = ensure_eval_attraction(conn, label)
        label_to_attr_id[label] = attr_id

        folder = os.path.join(DATASET_ROOT, "base", label)
        for image_path in list_images(folder):
            # 去重：同一评测文件已入库则跳过
            existed = conn.execute(
                text(
                    """
                    SELECT id FROM attraction_images
                    WHERE file_path = :file_path AND upload_by = :upload_by
                    LIMIT 1
                    """
                ),
                {"file_path": image_path, "upload_by": UPLOAD_BY},
            ).fetchone()
            if existed:
                inserted.append(
                    InsertedBaseImage(
                        image_id=int(existed.id),
                        label=label,
                        file_path=image_path,
                        attraction_id=attr_id,
                    )
                )
                continue

            emb = extract_embedding(image_path)
            new_id = insert_attraction_image(
                conn=conn,
                attraction_id=attr_id,
                file_path=image_path,
                upload_by=UPLOAD_BY,
                embedding=emb,
            )
            inserted.append(
                InsertedBaseImage(
                    image_id=new_id,
                    label=label,
                    file_path=image_path,
                    attraction_id=attr_id,
                )
            )

    return label_to_attr_id, inserted


def build_attraction_label_map(conn) -> Dict[int, str]:
    rows = conn.execute(
        text("SELECT id, name FROM attractions WHERE name LIKE '[MM_EVAL] %'")
    ).fetchall()
    out: Dict[int, str] = {}
    for r in rows:
        # name: [MM_EVAL] xxx
        label = str(r.name).replace("[MM_EVAL] ", "", 1)
        out[int(r.id)] = label
    return out


def evaluate_queries(conn, attraction_label_map: Dict[int, str]) -> Dict:
    details = []
    metrics = {
        "normal": {"total": 0, "top1_hit": 0, "top3_hit": 0},
        "hard": {"total": 0, "top1_hit": 0, "top3_hit": 0},
        "negative": {"total": 0, "top1_is_negative_like": 0},
    }

    # negative 这里按“top1是否落在animal/vehicle”做一个可解释检查
    negative_like_labels = {"animal", "vehicle"}

    for split in QUERY_SPLITS:
        folder = os.path.join(DATASET_ROOT, "query", split)
        for img_path in list_images(folder):
            emb = extract_embedding(img_path)
            rows = search_similar_images(conn, emb, top_k=TOP_K, exclude_image_id=None)

            ranked = []
            for r in rows:
                label = attraction_label_map.get(int(r.attraction_id), "external")
                ranked.append(
                    {
                        "image_id": int(r.id),
                        "attraction_id": int(r.attraction_id),
                        "attraction_name": str(r.attraction_name),
                        "label": label,
                        "distance": float(r.l2_distance),
                        "file_path": str(r.file_path),
                    }
                )

            top1_label = ranked[0]["label"] if ranked else None
            top3_labels = [x["label"] for x in ranked[:3]]

            expected_label = None
            img_name = os.path.basename(img_path)
            if "great_wall" in img_name or "temple" in img_name:
                expected_label = "architecture"
            elif "landscape" in img_name or "mountain" in img_name:
                expected_label = "nature"

            hit_top1 = expected_label is not None and top1_label == expected_label
            hit_top3 = expected_label is not None and expected_label in top3_labels

            if split in ("normal", "hard"):
                metrics[split]["total"] += 1
                metrics[split]["top1_hit"] += int(bool(hit_top1))
                metrics[split]["top3_hit"] += int(bool(hit_top3))
            else:
                metrics[split]["total"] += 1
                metrics[split]["top1_is_negative_like"] += int(
                    bool(top1_label in negative_like_labels)
                )

            details.append(
                {
                    "split": split,
                    "query_image": img_path,
                    "expected_label": expected_label,
                    "top1_label": top1_label,
                    "top3_labels": top3_labels,
                    "hit_top1": bool(hit_top1),
                    "hit_top3": bool(hit_top3),
                    "ranked_results": ranked,
                }
            )

    # 额外统计 top1 label 分布
    top1_counter = Counter([d["top1_label"] for d in details if d["top1_label"]])

    return {
        "metrics": metrics,
        "top1_distribution": dict(top1_counter),
        "details": details,
    }


def write_reports(report: Dict):
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Multimodal Dataset Eval Report",
        "",
        f"- dataset_root: `{DATASET_ROOT}`",
        f"- top_k: {TOP_K}",
        "",
        "## Metrics",
        "",
    ]

    m = report["metrics"]
    for split in ["normal", "hard"]:
        total = m[split]["total"]
        t1 = m[split]["top1_hit"]
        t3 = m[split]["top3_hit"]
        t1_acc = (t1 / total) if total else 0.0
        t3_acc = (t3 / total) if total else 0.0
        lines.append(f"- {split}: total={total}, top1={t1}/{total} ({t1_acc:.2%}), top3={t3}/{total} ({t3_acc:.2%})")

    n_total = m["negative"]["total"]
    n_like = m["negative"]["top1_is_negative_like"]
    n_ratio = (n_like / n_total) if n_total else 0.0
    lines.append(f"- negative: total={n_total}, top1 in {{animal,vehicle}} = {n_like}/{n_total} ({n_ratio:.2%})")

    lines.extend([
        "",
        "## Top1 Label Distribution",
        "",
    ])
    for k, v in sorted(report["top1_distribution"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")

    lines.extend([
        "",
        "## Detail File",
        "",
        f"详单见 `{os.path.basename(REPORT_JSON)}`",
    ])

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    cfg = DBConfig()
    engine = create_engine(cfg.sqlalchemy_url, future=True)

    with engine.begin() as conn:
        label_map, inserted = seed_base_images(conn)
        attraction_label_map = build_attraction_label_map(conn)
        report = evaluate_queries(conn, attraction_label_map)

    report["seed_summary"] = {
        "labels": label_map,
        "seed_count": len(inserted),
        "upload_by": UPLOAD_BY,
    }

    write_reports(report)

    print("✅ Eval done")
    print(f"- seeded images: {report['seed_summary']['seed_count']}")
    print(f"- report json: {REPORT_JSON}")
    print(f"- report md: {REPORT_MD}")


if __name__ == "__main__":
    main()
