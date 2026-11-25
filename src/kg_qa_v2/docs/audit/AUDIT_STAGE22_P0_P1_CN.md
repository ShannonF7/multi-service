
# Stage 22 P0/P1 审计报告（中文）

## 本阶段目标

本阶段只实现两项闭环：

1. P0：收紧 Router、问题结构化和执行计划的职责边界。
2. P1：补齐候选澄清的保存、展示、恢复和继续执行。

未在本阶段重构任务型 Skill，也未引入新的 Agent。

## 一、输入输出审计

### Router

输入：

- 当前用户问题；
- 是否携带图片、目标模态；
- 当前领域；
- 当前节点、定位节点；
- 最近实体、最近问题、近期消息；
- 待确认候选。

模型允许输出：

- intent
- context_dependency
- confidence

旧字段仍在 RouteDecision 中保留，但只用于兼容旧调用方；执行开关由程序推导，不再信任模型直接输出。

### Question Understanding

槽位模型只抽取：

- targets
- attribute
- start/end
- time_range
- comparison_targets
- constraints
- scope
- question_rewrite

槽位模型不再拥有检索方式、图谱范围和模态开关的决定权。EntityReference[] 继续作为 QuerySpec 的正式输出。

### Plan Compiler

根据 intent、route、输入模态和约束确定 Graph/Text/Image/Path 执行步骤。即使旧模型返回互相冲突的 needs_graph=false、needs_image=true，编译器也不会直接照用。

## 二、澄清状态机审计

### 保存

候选无法唯一对齐时，服务保存：

- pending_mention
- pending_candidates
- pending_question
- pending_domain_id
- pending_query_spec
- pending_route

内存会话和 PostgreSQL 持久会话均复用原有 pending_clarification 存储字段。

### 恢复

下一轮输入先经过 resolve_pending_clarification：

- 第2个、二个：按上一轮候选序号选择；
- 二楼的、二层的：只在上一轮候选层级路径中匹配；
- 未匹配时不擅自选择，也不创建新节点。

匹配成功后，以 pending_question 作为原始问题继续理解，带着已选择的节点 ID进入原来的计划和检索阶段。

### 前端

A 端 SSE 最终结果透传 clarification 字段，并在小V消息中展示候选按钮。点击按钮会向原会话发送“第 N 个”，从而触发后端恢复，而不是绕过后端直接拼答案。

## 三、记忆审计

ContextResolver 现在读取：

- 最近 8 条结构化消息；
- 上一轮问题；
- 待确认状态；
- 最近实体和会话摘要。

这些内容会进入 Router/槽位结构化上下文。原始历史只做有限窗口读取，不把无限历史直接塞给模型。

## 四、测试结果

- QA v2 全量测试：80 passed
- 新增 Stage 22 P0/P1 测试：Router 字段边界、Plan Compiler 冲突字段、候选序号/层级恢复、恢复后带节点 ID继续检索、会话待确认持久化，共 8 passed
- 编译检查：src/kg_qa_v2 通过
- A 端 Python 代理测试：当前 lab420pro 运行环境未安装 Django，无法执行 Django 测试收集；已完成模板逻辑定点审计，改动仅涉及候选样式、候选渲染和按钮回填提交。

## 五、已知限制

1. 当前“二楼的”恢复只在上一轮候选中做确定性匹配，不会对新的全领域候选召回；这是为了避免澄清状态被重新污染。
2. 长期语义记忆的向量检索尚未接入本阶段主链；当前已接入会话摘要、近期消息和已有长期画像的存储边界。
3. Graph/Text/Image 仍沿用现有适配器命名，真正的 LocationSkill、PathSkill、CompareSkill 等任务型 Skill 属于下一阶段 P3，不在本阶段混入。

## 六、结论

P0 的职责边界已闭合，P1 的 Clarification Resume Edge 已建立。下一步应在真实接口上验证：

- “财务部在哪里？”返回候选；
- “二楼的”恢复上一问并返回二层节点；
- 用户点击候选按钮后 A 端仍走同一 session；
- 不带待确认状态时，“二楼的”不会误选旧候选。
