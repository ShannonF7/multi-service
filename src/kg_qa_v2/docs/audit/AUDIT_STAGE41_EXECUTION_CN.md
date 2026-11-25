# QA v2 第三阶段执行层审计（中文）

审计日期：2026-10-12  
审计范围：NEARBY 空间查询、视觉任务执行、节点绑定图片、任务 Skill 路由、确定性答案路径。

## 一、实现结论

本阶段已把 Stage 2 的结构化理解接入真实执行层：

1. NEARBY 不再复用普通 LIST/HIERARCHY 邻接查询，而是编译为 GRAPH(operation=nearby)。
2. 周边查询限定在当前领域的 PublishedEntity，读取目标节点和同领域有坐标的已发布节点，使用经纬度计算米级距离，按距离、名称、节点 ID 稳定排序。
3. VISUAL_APPEARANCE 和 IMAGE_RETRIEVAL 进入 VisualQuerySkill。实体已经对齐时，优先读取该节点的已发布关联图片；没有绑定图片时才回退到文本到图片向量检索。
4. IMAGE_IDENTIFICATION 仍进入 ImageRecognitionSkill，上传图片不会使用当前 GPS/会话节点替代图片识别结果。
5. SkillRegistry 增加请求类型优先级：先按 request_type 选具体任务 Skill，再按意图和步骤类型兜底。
6. NEARBY 有确定性答案组织和引用编号，不需要为“附近有什么”再调用答案大模型。

## 二、输入、输出与逻辑审计

### 1. NEARBY

输入：
- QuerySpec.requests[] 中存在 request_type=NEARBY
- 已对齐的 resolved_node_ids[0]
- 可选 radius、category

执行：
- PlanCompiler 生成唯一 GRAPH 步骤，参数为 operation=nearby
- PublishedGraphGateway 调用 get_published_nearby_nodes
- 服务端限制半径 1～5000 米，默认 300 米；返回数量限制 1～100，默认 20
- 结果证据内容包含“附近约 X 米处有：节点名”

输出：
- 按距离排序的图谱证据
- metadata.distance_m
- 确定性答案中的 [E#] 引用

安全边界：
- 只读 PublishedEntity
- 强制携带领域 ID
- 通过 QA gateway 时仍强制 published_only=True
- 不接受自由 Cypher

### 2. 视觉查询

输入：
- VISUAL_APPEARANCE 或 IMAGE_RETRIEVAL
- 已解析节点 ID

执行：
- 计划同时读取节点详情和图片步骤
- 图片步骤优先使用节点详情中的 root.assets
- 图片 Evidence 的 source_url 兼容 source_url/image_url/proxy_image_url/url
- 节点无绑定图片时，使用 image_query（目标名）回退到已有图片向量检索

输出：
- 图片 Evidence，包含 asset_id、node_id、source_url
- 由既有答案层返回最多 3 张图片
- 图片仍与实体节点绑定，不再使用当前 GPS 节点冒充上传图片结果

### 3. 上传图片识别

输入：
- image_path
- IMAGE_IDENTIFICATION 或兼容旧的 IMAGE 请求

执行：
- 继续使用已有图片向量检索
- 由已有生产函数扩大召回池并执行上传图片候选重排
- Top1 节点再读取已发布图谱详情

输出：
- “上传图片最相似的已发布图片对应节点为……”的证据
- 节点详情和图片证据可同时引用

## 三、测试结果

专项测试：
- test_stage41_execution.py：5 个用例通过
- Skill 注册回归：4 个用例通过
- 第三阶段相关回归合计：25 个用例通过

QA v2 全量：
- 149 passed, 1 warning
- 唯一 warning 为现有 Pydantic class-based config 弃用警告，不是本阶段引入。

语法和代码审计：
- 第三阶段涉及 Python 文件全部 py_compile 通过
- 本阶段涉及文件 git diff --check 通过
- 服务重启后 fastapi_travel.service=active
- http://127.0.0.1:7001/docs 返回 200

## 四、没有伪装成已完成的部分

1. 本阶段没有修改底层上传图片重排模型；现有上传图片路径已经调用已有的候选扩大召回和 _rerank_uploaded_image_candidates。
2. 节点绑定图片走的是已发布节点资产直接读取，不需要再次用向量模型判断“这个节点是不是自己”；这不是漏掉重排，而是已经完成实体对齐后的确定性绑定。
3. 如果视觉查询没有解析到实体节点，仍会走图片文本向量回退；该回退路径当前使用已有向量近邻结果，文本图片检索的独立 reranker 仍属于后续阶段。
4. 周边查询当前在 Python 中对同领域有坐标节点计算距离。大规模领域后续可改为 Neo4j 空间索引/包围盒预过滤，但目前逻辑正确、范围有限、结果可审计。

## 五、审计结论

本阶段没有新增针对“东门”或某个特殊节点的硬编码。任务分支由 request_type、实体对齐结果和统一执行计划决定：

Question Understanding → QuerySpec.requests → PlanCompiler → SkillRegistry → Tool → Evidence → 确定性/模型答案

因此：
- “东门长啥样”走视觉查询；
- “东门附近有什么”走空间查询；
- “这张图片是哪里”走上传图片识别；
- 同一套逻辑可用于其他已发布节点。

