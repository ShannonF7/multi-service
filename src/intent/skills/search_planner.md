# 检索范围规划器 (Search Planner)

**注意：当前作为参考逻辑存在，实际执行逻辑已在 `langgraph_agent.py` 的 `planner_node` 中基于规则硬编码实现。**

## 逻辑规则

输入：
- 意图话题 (Topic)
- 语义分析 (Semantic Analysis: Subject, Attributes, Scope Hint)
- 当前位置 (Current Space ID)

### 规则一：无检索 (No Search)
若 话题 == NONE 或 类型 == CHAT
→ 模式: `NO_SEARCH`

### 规则二：全局暗示 (Global Semantics)
若 语义暗示范围 (Scope Hint) 包含 "Global"
→ 模式: `GLOBAL` (全库搜索)

### 规则三：百科 (Encyclopedia)
若 话题 == 百科
→ 模式: `GLOBAL`

### 规则四：特定领域 (Domain: 建筑/历史/...)
若 话题属于 [建筑, 军事, 历史, 信仰, 民俗]:
- 默认策略: `LOCAL_HIERARCHY` (当前位置 + 父级)
- *注：若能从语义明确提取出非当前位置的实体（如在“古堡”问“故宫”），规划器应能重定向 target_space_ids（此逻辑尚待完善）。*

### 规则五：服务与路线 (Service & Route)
若 话题 == 服务 或 路线
→ 模式: `SCENIC_GLOBAL` (景区全域)
→ *理由：用户可能在找附近的厕所，或者问另一个景点的路。*
