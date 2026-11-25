# Stage 16：B 端主应用与 HTTP/SSE 接入审计

## 变更范围

- 新增 src/kg_qa_v2/api.py，提供 /kg-qa-v2/query 与 /kg-qa-v2/stream。
- 主 app.py 同时挂载旧版 /kg-qa/* 和新版 /kg-qa-v2/*，没有覆盖旧实现。
- SSE 使用真正的标准换行帧：event: name\ndata: json\n\n。
- A 端后续通过代理把现有页面协议转换到 v2，避免页面直接理解两套协议。
- GPS 请求先在当前领域的已发布节点中解析最近节点；解析失败仍保留经纬度，不会伪造节点。

## 输入审计

- QARequest 复用现有字段校验：领域、问题、会话、用户位置、当前节点、图片输入和证据数量。
- session_id 优先使用请求会话，其次兼容 conversation_id。
- has_image 只表示请求是否带图片；没有 image_path 时不会伪造图片文件。

## 输出审计

- 同步接口返回 QAV2Service 的结构化结果。
- 流式接口逐事件输出 started/process/answer_started/token/completed/error。
- 不输出模型私有思维链，只输出可审计阶段状态和答案 token。
- completed 事件包含最终答案、引用、图片、警告和总耗时。

## 逻辑与副作用

- 路由层只做请求转换、位置解析和线程队列，不负责实体识别、检索或答案生成。
- v2 服务构造默认连接本机 7020 Qwen 服务、已发布图谱、已有文本/图片检索服务。
- 每个 SSE 请求创建一个守护线程；队列结束后发送结束哨兵，避免请求线程悬挂。
- 主应用日志中对 /kg-qa-v2/stream 跳过请求体读取，避免 BaseHTTPMiddleware 破坏 SSE 请求体。

## 测试结果

- pytest -q src/kg_qa_v2：58 passed。
- py_compile app.py src/kg_qa_v2/api.py src/kg_qa_v2/service.py：通过。
- 导入主应用并枚举路由：确认存在 /kg-qa-v2/query、/kg-qa-v2/stream。
- 修复了原 SSE 编码测试的错误预期，现验证的是浏览器可消费的真实换行。

## 风险与边界

- 远程服务由全局 SecurityMiddleware 统一鉴权；A 端必须继续携带现有 RAG_SYNC_API_KEY。
- 位置解析依赖图谱数据库不可用时会降级为仅传坐标；问答服务仍可继续按其他上下文运行。
- 旧路由暂不删除，待 A 端和真实生产流式验证通过后再考虑下线。

## 闲聊边界复核

- CASUAL 路由不执行图谱、文本或图片检索。
- CASUAL 过程不再展示“证据整理”阶段，A 端会将其标记为聊天消息。
- 真实请求“你好”返回 200，route 和 intent 均为 CASUAL，evidence 为空。
