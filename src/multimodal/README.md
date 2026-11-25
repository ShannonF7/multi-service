# 多模态图像模块

本目录包含图像编码模型、评估数据结构和离线实验脚本。当前维护的评估主线为 expanded_v3。

## 目录职责

- model_registry.py、encoders.py、clip_feature_extractor.py：模型注册与图像编码器。
- schemas.py、metrics.py、node_aggregation.py：评估数据结构、检索指标和节点聚合。
- scripts/：按审计、基准、数据集、评估、OCR、复核、报告和检索分类的 CLI 脚本。
- docs/：维护中的实验说明和旧 attractions_db PoC 文档。
- image_retrieval_pipeline.py：保留的旧数据库 CLI 入口，供现有脚本和文档继续使用。

## 运行方式

从项目根目录运行，例如：

    PYTHONPATH=. python -m src.multimodal.scripts.audits.asset_audit --source-scenic-id 4
    PYTHONPATH=. python -m src.multimodal.scripts.evaluations.evaluate_simclr_e0 --help

已文档化的旧命令入口保留为兼容转发模块。

## HTTP 接口约定

当前目录没有 FastAPI router 或 HTTP endpoint，因此这里不定义统一 HTTP 响应包装。若后续新增接口，应遵循项目统一响应模型、错误码、异常处理及请求响应日志规范。
