# QA V2 Stage 40：附近与视觉问题理解审计

## 本阶段目标

让 Question Understanding 能区分：

- 普通位置：LOCATION；
- 周边检索：NEARBY；
- 实体外观：VISUAL_APPEARANCE；
- 已发布图片请求：IMAGE_RETRIEVAL；
- 上传图片识别：IMAGE_IDENTIFICATION。

同时增加高置信度 Fast Gate，避免“东门长啥样”因为 Router 不认识口语表达而进入本地 7B fallback。

## 输入处理

本阶段支持以下表达：

- “东门附近有什么”“东门周边有什么”；
- “东门长啥样”“东门长什么样”“东门外观如何”；
- “给我看看东门照片”；
- “这张照片是哪里”。

模型返回的 requests 会兼容 type 和 request_type 两种字段写法，避免结构化模型使用用户方案中的 type 字段时整体降级。

## 输出

QuerySpec 现在可以产生：

    NEARBY
    VISUAL_APPEARANCE
    IMAGE_RETRIEVAL
    IMAGE_IDENTIFICATION

同时生成：

    response_requirements.text
    response_requirements.images
    response_requirements.map

视觉请求会设置 needs_image=true；普通位置请求不会因为存在当前实体就自动要求图片。

## Router 行为

明确的视觉和附近表达会走确定性 fallback，不再默认调用 7B：

- “附近有什么” → LIST 粗路由 + NEARBY 请求；
- “长啥样” → IMAGE 粗路由 + VISUAL_APPEARANCE 请求；
- “给我看看照片” → IMAGE 粗路由 + IMAGE_RETRIEVAL 请求；
- 上传图片询问位置 → IMAGE 粗路由 + IMAGE_IDENTIFICATION 请求。

这里的 LIST/IMAGE 只是兼容当前 RouteDecision 的粗粒度路由；最终任务类型保存在 requests 中，避免让 Router 承担过多细分类别。

## 当前未完成边界

本阶段只改理解层，没有改执行层：

1. PlanCompiler 尚未为 NEARBY 生成空间距离查询；
2. NearbySkill 尚未接入；
3. VisualQuerySkill 尚未接入；
4. IMAGE_IDENTIFICATION 尚未在本阶段改变已有图片向量检索和重排策略；
5. response_requirements 尚未完全驱动前端图片/地图展示；
6. 7B fallback 仍然存在于未命中本阶段 Fast Gate 的复杂问题中。

因此，本阶段验收标准是“结构化理解正确”，不是“最终页面已经返回附近节点或图片”。

## 副作用审计

- 不访问图谱数据库；
- 不执行空间计算；
- 不调用答案 API；
- 不写入会话、图谱或向量库；
- 修改仅涉及 Router/Question Understanding/slots/prompts 以及测试；
- 未写入任何密钥。

## 测试结果

- 第二阶段专项测试：5 passed；
- 理解、Router、实体引用回归：23 passed；
- QA V2 全量测试：144 passed，1 warning；
- warning 为既有 Pydantic v2 class-based config 弃用提示。

## 结论

第二阶段完成。理解层已经准备好把附近和视觉任务交给不同 Skill，下一阶段开始改 PlanCompiler、NearbySkill 和 VisualQuerySkill。