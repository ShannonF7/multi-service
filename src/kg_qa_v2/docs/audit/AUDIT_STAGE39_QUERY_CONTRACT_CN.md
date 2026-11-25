# QA V2 Stage 39：QuerySpec 任务与输出模态契约审计

## 本阶段目标

为“附近”和“视觉外观/图片检索/上传图片识别”建立稳定的数据契约。本阶段只扩展契约，不改变 Router、PlanCompiler、Skill 或实际检索行为。

## 输入

新增 QueryRequest 类型：

- NEARBY
- VISUAL_APPEARANCE
- IMAGE_RETRIEVAL
- IMAGE_IDENTIFICATION

新增请求字段：

- target：请求针对的对象；
- radius：空间请求的半径，当前只作为结构化字段保存；
- category：附近对象的类别约束；
- constraints：保留原有自由约束字段。

新增 QuerySpec.response_requirements：

- text：是否需要文字回答，默认 true；
- images：是否需要图片，默认 false；
- map：是否需要地图/空间展示，默认 false。

## 输出与兼容性

旧代码仍可以使用：

- QuerySpec(intent="LOCATION", targets=[...])；
- QueryRequest(request_type="ATTRIBUTE", attribute=...)；
- 旧的 needs_graph、needs_text、needs_image、needs_path 字段。

新字段具有安全默认值，不会改变旧问题的执行计划，也不会让现有问题自动触发图片或地图检索。

## 逻辑审计

本阶段没有把“附近”或“长啥样”直接写成执行规则，避免契约和业务逻辑混在一起。

当前阶段的实际效果是：

1. Pydantic 可以合法表示 NEARBY 和视觉类请求；
2. 后续 Router/Understanding 可以输出 response_requirements；
3. PlanCompiler 尚未消费这些新类型；
4. NearbySkill、VisualQuerySkill 尚未接入；
5. 旧的 IMAGE 意图仍然保持原行为。

## 副作用审计

- 不访问数据库；
- 不调用模型；
- 不修改图谱、向量库、会话或环境配置；
- 只增加 Python 数据模型和单元测试；
- 不包含任何用户密钥或外部服务写操作。

## 测试结果

- 新增契约测试及旧的阶段 31/34 测试：5 passed；
- QA V2 全量测试：139 passed，1 warning；
- warning 为既有 Pydantic v2 class-based config 弃用提示，不是本阶段失败。

## 阶段结论

本阶段完成。下一阶段才开始修改 Question Understanding：将 NEARBY、VISUAL_APPEARANCE、IMAGE_RETRIEVAL、IMAGE_IDENTIFICATION 从自然语言中结构化抽取出来，并生成 response_requirements。