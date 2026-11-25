# QA v2 阶段 21：Router + Skill 审计报告

审计时间：2026-10-05
审计范围：src/kg_qa_v2/skill_router.py、retrieval.py、__init__.py、阶段 21 测试
审计目标：将 Graph、Text、Image 检索正式包装为确定性 Skill，并由 SkillRouter 统一选择、执行和返回结果；不引入全量 Multi-Agent。

## 一、架构结论

本阶段后的实际链路：

Context Resolver
→ Router
→ Question Understanding
→ Entity Resolution
→ Plan Compiler
→ SkillRouter
→ GraphSkill / TextSkill / ImageSkill
→ Evidence Fusion
→ Answer / Citation / SSE

其中：

- Router 只判断问题类型；
- Entity Resolution 统一解析实体，Skill 不重复解析实体；
- PlanCompiler 只生成确定性 ExecutionPlan；
- SkillRouter 只根据 ExecutionPlan 选择能力；
- Graph、Text、Image Skill 不自主规划，不调用工具链，不生成 Cypher；
- 多路 Skill 的并行执行使用线程池；
- PlanExecutor 保留为兼容入口，内部已经委托给 SkillRouter。

## 二、输入输出审计

### SkillContext

输入：

- domain_id；
- 当前问题；
- QuerySpec；
- ContextSnapshot；
- 当前 PlanStep；
- published_only。

约束：

- published_only 在 Router 边界固定为 True；
- Skill 不能自行切换领域；
- Skill 不接收未解析的自然语言实体作为自主决策入口，而是消费已经完成问题理解和实体对齐后的计划。

### SkillResult

输出：

- skill_name：GraphSkill、TextSkill 或 ImageSkill；
- step_type：GRAPH、TEXT 或 IMAGE；
- ok：是否成功；
- evidence：统一的证据字典列表；
- warnings：本 Skill 的失败和警告；
- metadata：执行范围等审计信息。

如果旧检索端口返回 Pydantic Evidence，也会在 Skill 边界转换成字典；PlanExecutor 再统一转换为 QA v2 Evidence。

## 三、执行逻辑审计

### GraphSkill

复用现有 Graph retriever，负责已发布图谱节点、关系、层级、路径等查询。

### TextSkill

复用现有领域文本混合检索服务，负责文本补充证据。

### ImageSkill

复用现有图片文本检索和上传图片检索服务，负责图片证据。

### SkillRouter

1. 过滤掉非 Graph/Text/Image 的计划步骤；
2. 按步骤映射到对应 Skill；
3. 当计划标记 parallel 且有多个 Skill 时并行执行；
4. 单个 Skill 失败时保留其他 Skill 结果；
5. 为失败 Skill 返回可审计 warning；
6. 不改变 EvidenceFusion 的去重、排序和数量限制。

### PlanExecutor 兼容层

原有调用方式不变：

PlanExecutor.execute(...)
→ SkillRouter.execute(...)
→ Evidence 转换
→ EvidenceFusion
→ RetrievalResult

因此现有 Service、Pipeline 和测试不需要改调用方式。

## 四、测试记录

阶段 21 专项测试：

- Skill 名称和计划选择；
- published_only 固定为 True；
- Graph/Text/Image 三路并行执行；
- 单个 Skill 失败不影响其他 Skill；
- SkillContext 和 SkillResult 契约。

结果：

- 5 passed

QA v2 全量回归：

- 79 passed
- 1 个既有 Pydantic 弃用警告
- 未发现阶段 21 引入的失败。

语法检查：

- compileall 通过。

## 五、边界与后续建议

本阶段没有引入：

- Graph Agent；
- Text Agent；
- Image Agent；
- 自由 Planner Agent；
- Multi-Agent 协调；
- 自动循环重规划。

这不是缺失，而是有意保持 Skill 的确定性。只有未来出现“证据不足时需要重新决定检索方向”的复杂问题，才应在 SkillRouter 外层增加有限的 Evidence Judge + Re-plan 分支。

当前 SkillRouter 已经完成能力模块化，但 Skill 还没有单独的业务数据库状态，也没有自主记忆。上下文、实体状态和会话记忆仍由前置 Memory/Context 层负责，这是正确的职责边界。
