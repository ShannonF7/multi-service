# QA v2 阶段 15 审计：四层记忆管理器

## 1. 本阶段目标

补齐记忆系统的独立模块，避免每轮把所有历史消息直接塞入模型。

当前分层：

- Working Memory：本轮问题、领域、位置、图片输入；
- Entity State：当前实体、当前节点、最近实体、待确认候选；
- Session Memory：原始会话消息；
- Session Summary：压缩后的主题、目标和已确认实体；
- Long-term Semantic Memory：用户跨会话的稳定偏好或语义记忆。

## 2. 新增接口

新增 memory_manager.py：

- MemoryStore：Redis/PostgreSQL 等生产存储实现的最小协议；
- MemoryManager.resolve：按需读取 session 和长期记忆；
- MemoryManager.update_after_turn：保存用户/助手一轮消息；
- MemoryUpdateResult：返回压缩状态和展示事件。

QA v2 不在该模块内直接连接 Redis 或 PostgreSQL，避免把存储实现和问答逻辑耦合。

## 3. 会话压缩规则

当满足任一条件时触发压缩：

- 消息轮数超过 max_messages；
- 消息总字符数超过 max_chars。

压缩时：

- 旧消息交给 summarizer；
- 保留最近 keep_recent_messages 条；
- 保存 session_summary.summary；
- 增加摘要版本；
- 返回 memory_compressed 展示事件。

如果未注入摘要模型，则使用保守的确定性摘要，不伪造新的事实。

## 4. 长期记忆读取规则

- 默认不读取长期记忆；
- 只有 need_long_term=True 且存在 user_id 时读取；
- 存储层调用被限制为最多 10 条；
- 读取后再次按当前 domain_id 过滤；
- 不把其他领域的长期记忆注入当前问题。

## 5. 输入输出审计

输入：

- session_id、user_id、domain_id；
- 当前问题；
- current_location、has_image；
- 是否需要长期记忆；
- 存储层返回的历史记录。

输出：

- MemoryContext；
- ContextSnapshot；
- MemoryUpdateResult；
- 可直接推送给前端的压缩事件。

## 6. 测试结果

执行：

- python -m pytest -q src/kg_qa_v2
- python -m compileall -q src/kg_qa_v2

结果：

- 54 passed；
- 编译检查通过。

测试覆盖：

- 长期记忆按需读取；
- 长期记忆领域隔离；
- 会话消息追加；
- 超阈值压缩；
- 最近消息保留；
- 自定义摘要；
- 实体状态保留；
- 压缩事件返回。

## 7. 当前边界

- 尚未把 MemoryManager 注入主 QAV2Service；
- 尚未实现生产 Redis/PostgreSQL MemoryStore；
- 尚未把异步 Memory Update 接到接口响应之后；
- 当前摘要模型是可注入的，默认确定性摘要只用于安全降级。

## 8. 审计结论

本阶段通过。记忆模块的接口与压缩语义已明确，可以在主 app 接入时绑定实际 session 存储和异步写回。
