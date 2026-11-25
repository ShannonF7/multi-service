# QA v2 行为回归集

这里保存真实模型接入后的 Gold Case，不把它们混入普通单元测试。

建议文件：

- `router_gold.jsonl`：Jev intent 和 routing flags。
- `slot_gold.jsonl`：Qwen 槽位和问题改写。
- `entity_alignment_gold.jsonl`：候选选择、AMBIGUOUS、UNRESOLVED。
- `planning_gold.jsonl`：QuerySpec 到 ExecutionPlan 的预期。

每条样本至少包含：

```json
{
  "case_id": "tyut-location-001",
  "domain_id": "tyut",
  "question": "财务部在哪里？",
  "context": {},
  "expected": {}
}
```

真实模型评估需要记录：准确率、置信度、错误类型、延迟、输入 token、输出 token 和成本。
阈值调整必须重新跑这批数据。
