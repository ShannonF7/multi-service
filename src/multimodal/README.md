# multimodal（独立目录版）

这个目录只面向你指定的数据库：`attractions_db`，并且只操作这两张表：

- `attractions`
- `attraction_images`

## 已交付内容

- `db_schema_attractions.md`：数据库结构与关系说明
- `image_retrieval_pipeline.py`：新图入库 + 以新图做相似检索（128维）

## 设计约束（与你当前库保持一致）

- 当前 `attraction_images.embedding` 是 `vector(128)`。
- 因此本脚本复用项目内现有特征提取器，输出 `128` 维向量并写入 pgvector。
- 脚本默认连接你提供的库：`localhost / zhangbi_user / attractions_db`。

## 使用方式

在项目根目录执行（确保 Python 环境可用，并且模型权重文件存在于项目既有位置）：

```bash
python -m src.multimodal.image_retrieval_pipeline \
  --image-path /absolute/path/to/new_image.jpg \
  --attraction-id 1 \
  --upload-by zhangbi
```

可选参数：

- `--top-k 10`：返回前10个相似结果
- `--storage-dir /some/path`：图片副本存储目录
- `--db-host --db-port --db-user --db-password --db-name`：数据库连接覆盖

## 输出说明

运行后会输出：

1. 新插入图片的 `id`
2. 存储路径
3. 相似检索结果（含距离 `l2_distance`，越小越相似）

## 注意

- 若库里很多历史图片 `embedding` 为空，相似结果可能较少；可先批量补齐 embedding。
- 若你后续坚持改成标准 CLIP 全维（如512），需要同步做数据库字段迁移。当前实现不改库结构，避免影响现网。
