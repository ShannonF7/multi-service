# attractions_db 表结构与关系说明（multimodal）

> 连接信息（你提供）：`psql -h localhost -U zhangbi_user -d attractions_db`
>
> 核对时间：2026-04-13

## 结论速览

- 当前库中**没有**名为 `images` 的表。
- 实际用于图片与向量检索的是：`attraction_images`。
- 景点主表是：`attractions`。
- 关系是：`attraction_images.attraction_id -> attractions.id`（`ON DELETE CASCADE`）。

---

## 1) `attractions`（景点主表）

| 字段 | 类型 | 可空 | 默认值 | 说明 |
|---|---|---|---|---|
| `id` | `integer` | 否 | `nextval('attractions_id_seq')` | 主键 |
| `name` | `varchar(255)` | 否 | - | 景点名称 |
| `description` | `text` | 是 | - | 景点描述 |
| `position` | `varchar(50)` | 是 | - | 位置 |
| `jpg_path` | `text[]` | 是 | - | 图片路径数组 |
| `upload_by` | `varchar(255)` | 否 | - | 上传人 |

索引：
- `attractions_pkey` (`id`)

---

## 2) `attraction_images`（景点图片向量表）

| 字段 | 类型 | 可空 | 默认值 | 说明 |
|---|---|---|---|---|
| `id` | `integer` | 否 | `nextval('attraction_images_id_seq')` | 主键 |
| `attraction_id` | `integer` | 否 | - | 外键，关联景点 |
| `file_path` | `text` | 否 | - | 图片路径（相对或绝对） |
| `upload_by` | `varchar(255)` | 否 | - | 上传人 |
| `embedding` | `vector(128)` | 是 | - | 向量特征（用于相似检索） |

索引：
- `attraction_images_pkey` (`id`)

外键：
- `attraction_images_attraction_id_fkey`
  - `attraction_images.attraction_id` → `attractions.id`
  - 删除行为：`ON DELETE CASCADE`

---

## 3) 关系图（ER 简版）

```text
attractions (1)  ───────────────< (N) attraction_images
   id (PK)                           id (PK)
                                     attraction_id (FK -> attractions.id)
                                     file_path
                                     upload_by
                                     embedding vector(128)
```

---

## 4) 对“CLIP检索”的落地影响

当前 `attraction_images.embedding` 维度是 `128`，因此：

- 直接使用标准 CLIP 常见向量（如 `512`）会与库字段不匹配。
- 当前项目已存在可直接匹配 `vector(128)` 的图像特征提取流程（`src/cv/feature_extractor.py`）。
- 若必须切到 CLIP 全维向量，需要额外做库结构迁移（例如改为 `vector(512)`）或加入降维方案。

本目录下已提供独立脚本 `image_retrieval_pipeline.py`，默认按当前库结构使用 `128` 维流程，避免破坏现有库。
