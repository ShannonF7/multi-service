# 本地决策模型部署约定

## 接口

`LocalDecisionClient` 对接 OpenAI-compatible 的：

```text
POST {LOCAL_DECISION_BASE_URL}/chat/completions
```

模型只需要返回 JSON。当前路由、实体对齐和引用校验都通过 `DecisionModel.decide(task, payload)` 进入，不允许模型直接访问图谱。

## 环境变量

```bash
DECISION_MODEL_PROVIDER=local
LOCAL_DECISION_BASE_URL=http://127.0.0.1:7020/v1
LOCAL_DECISION_MODEL=Qwen2.5-7B-Instruct-AWQ
LOCAL_DECISION_TIMEOUT=60
# 如果本地服务开启鉴权，再配置 LOCAL_DECISION_API_KEY
```

`DECISION_MODEL_PROVIDER=auto` 时，只要本地地址和模型同时存在就优先本地；否则尝试已有 Hosted Jev 配置。没有任何配置时，问答理解层使用确定性规则回退。

## 进程内 Transformers 模式

如果不想单独启动 vLLM，可使用：

```bash
DECISION_MODEL_PROVIDER=local
LOCAL_DECISION_BACKEND=transformers
LOCAL_DECISION_MODEL_PATH=/models/Qwen3.5-4B
LOCAL_DECISION_DEVICE_MAP=auto
LOCAL_DECISION_MAX_NEW_TOKENS=256
```

该模式第一次请求时才加载 tokenizer 和模型，`device_map=auto` 交给 Transformers 根据可见 GPU 分配。生产环境应先单独用目标权重验证显存、并发和生成格式，再接入问答主链。

## GPU 部署边界

L40×2 可以作为后续本地推理部署资源，但模型权重、量化方式、张量并行参数和服务端口需要根据实际 Qwen 权重版本单独验证。部署前可用 `inspect_transformers_runtime()` 做文件预检；本次已在独立环境中完成 vLLM 安装和真实启动验证，未修改已有向量化、重排服务。

部署前可调用 `inspect_transformers_runtime()` 检查模型目录是否至少包含 `config.json` 和权重文件。检查通过后仍需进行真实加载和问答质量评估。

## 当前已验证部署

实际模型目录：

```text
/home/zhangbi/Zhangbi_Traveler/multimodal/qa_qwen2_5_7b_awq
```

vLLM 服务地址为 `127.0.0.1:7020`，独立环境为：

```text
/home/zhangbi/txweb/envs/kg_qa_vllm
```

服务使用 Qwen2.5-7B-Instruct AWQ、2048 上下文长度和 lm-format-enforcer JSON 约束后端。它只承担路由、候选对齐和引用校验，不访问图谱、不替代检索器。
