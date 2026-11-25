# QA v2 第四阶段审计：统一图片重排

审计日期：2026-10-12  
审计范围：文本找图片、节点视觉查询、上传图片识别的候选池和 VL 重排。

## 1. 修改内容

新增统一入口：
- src/rag/service/image_embedding_service.py
- rerank_image_candidates(query, items, instruction, timeout)

该函数：
1. 接收文本查询或上传图片查询；
2. 将候选图片转换为 Qwen-VL Reranker 所需的 image/title 文档；
3. 记录 rerank_required、rerank_status、rerank_method；
4. 将向量分数与 VL 分数按 0.55/0.45 融合；
5. 按融合分数降序返回；
6. 模型服务失败、候选缺少 URL、返回数量不一致时保留原召回顺序并写入状态，不伪造重排成功。

原有上传图片函数已改为复用该统一入口，保持“上传图像 + 候选图片”的专门指令。

## 2. 三条图片链路

### 文本找图片

输入：
- 用户文本需求；
- 当前领域；
- 图片候选数量。

处理：
- 默认向量召回池扩大为 requested*4，至少 12、最多 20；
- 召回结果超过 1 个时调用 VL 重排；
- 重排后截取最终 Top-K。

### 节点视觉查询

输入：
- 已对齐节点；
- 节点绑定的已发布图片；
- 用户视觉需求。

处理：
- 先读取该节点的已发布资产；
- 多张图片时使用用户视觉需求进行 VL 重排；
- 再返回 Top-K。

注意：这里不是让模型重新判断“图片属于哪个节点”。节点已经通过实体对齐确定，重排只负责同一节点图片之间的相关性排序。

### 上传图片识别

输入：
- 用户上传图片；
- 当前领域。

处理：
- 图片向量召回候选；
- 现有实现扩大候选池；
- 统一 VL 重排；
- 按融合分数确定 Top1/TopK；
- 再读取 Top1 对应的已发布图谱节点。

上传图片不会使用 GPS 或会话节点替代识别结果。

## 3. 输入输出审计

成功输出包含：
- asset_id
- node_id（如果有）
- source_url
- vector_score
- image_rerank_score
- score
- rerank_status
- rerank_method

异常输出：
- rerank_status=failed / skipped_missing_source_url
- 保留原向量召回顺序
- 不把失败标记伪装成完成

单候选情况：
- 不触发模型重排，因为不存在候选间排序问题；
- 保留候选原始结果，避免无意义延迟。

## 4. 测试结果

专项测试：
- 图片重排专项及原图片适配器：8 passed
- QA v2 全量：152 passed，1 warning
- warning 为既有 Pydantic class-based config 弃用警告。

代码检查：
- image_embedding_service.py 和 retrieval_adapters.py py_compile 通过；
- 本阶段涉及改动无 diff 空白错误。

## 5. 当前边界

1. 重排服务仍然是 fail-open：服务不可用时不会阻塞整个问答，但会明确记录失败状态。
2. 节点绑定图片的前提是节点已经完成实体对齐；实体未对齐时仍会走图片向量回退。
3. 当前融合权重为代码中的稳定默认值 0.55/0.45，后续可通过离线东门/南门样本集校准，而不是针对单个地点硬编码。
4. 当前只保证问答执行层使用重排；还没有增加一个管理后台页面来展示每次重排的详细候选轨迹。

