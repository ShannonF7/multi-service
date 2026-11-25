# KG-QA v2 第 18 阶段审计：分阶段耗时、按需定位与图片输入

## 1. 本阶段目标

本阶段针对问答页面“长时间停留在准备回答”、普通问题被 GPS 阻塞、无法展示具体耗时以及图片入口未形成真实输入链路的问题进行审计和修复。

## 2. 代码位置与数据流

### 2.1 闲聊早退

- router.py：对明确问候、感谢、告别、自我介绍等固定社交语句做快速 CASUAL 判定，不调用本地决策模型。
- understanding.py：CASUAL 不再调用槽位抽取模型。
- pipeline.py：理解结果为 CASUAL 后直接生成闲聊计划并返回，不进入候选实体、图谱、文本、图片检索。
- service.py / streaming.py：闲聊仅调用答案生成器；无证据时不执行引用校验，也不产生虚假引用。

输入：用户文本、领域上下文、图片标记。  
输出：route=CASUAL、无 evidence/citations、可流式返回的聊天答案。  
边界：固定社交语句之外的复杂闲聊仍可经过 Router，但仍会在 pipeline 的 CASUAL 分支终止检索链路。

### 2.2 阶段耗时

pipeline.py 和 understanding.py 记录：

- context_ms
- router_ms
- slot_ms
- alignment_ms
- plan_ms
- retrieval_ms
- answer_ms
- answer_first_token_ms
- total_ms

同步接口在最终响应的 timings 返回；流式接口在 completed 事件返回，首 token 到达后写入 answer_first_token_ms。

审计结论：各阶段字段由程序计时，不由模型估算。缺少的阶段表示该阶段按计划未执行，例如 CASUAL 不应出现 retrieval_ms。

### 2.3 图片上传

api.py 增加 POST /kg-qa-v2/upload：

- 输入：multipart 字段 file，要求 image/*。
- 限制：空文件拒绝，最大 10 MB，文件名只保留 basename，扩展名限制为 jpg/jpeg/png/webp/gif/bmp。
- 输出：临时文件绝对路径 image_path、原文件名、MIME、字节数。
- 后续：A 端把该路径作为 image_path 写入问答请求，target_modality=IMAGE；检索适配器只有在路径真实存在时才调用图片检索。

安全边界：上传接口不把路径直接当作可执行内容，不允许调用方指定目标文件名。当前临时文件生命周期由部署环境清理策略负责，后续可增加定时清理。

## 3. 流式链路审计

api.py 的 /stream 在线程中消费 service.stream_ask()，队列即时转发事件；服务层会先发送 started，再发送上下文、理解、规划、检索等 process 事件，答案生成阶段发送 token，结束发送 completed。

因此页面不再等待最终答案才收到第一条可见事件。A 端还需继续把这些事件映射为页面 trace。

## 4. 测试结果

- src/kg_qa_v2：67 passed，1 个既有 Pydantic 弃用警告。
- 覆盖：CASUAL 不调用决策/槽位模型、pipeline 早退、阶段计时字段、上传文件保存、请求参数映射、GPS 解析。
- 编译检查：python -m compileall -q src/kg_qa_v2 通过。

## 5. 未解决事项

- 复杂知识问答的模型推理耗时仍可能较长，必须以线上 router_ms、slot_ms、retrieval_ms、answer_first_token_ms 为依据继续优化。
- 上传文件当前存临时目录；生产环境应配置可清理目录和定时清理策略。
- 图片能否命中领域图片库，取决于图片嵌入服务和该领域已发布图片索引，不应把“上传成功”误认为“已检索到图片证据”。

## 6. 引用标记兜底

答案校验层现在会清理没有对应证据的 E 编号以及 graph/text/image evidence id；A 端渲染层也会再次清理未被证据映射的内部标记，避免页面出现 graph:...、[E2] 但引用来源为 0 的不一致结果。
