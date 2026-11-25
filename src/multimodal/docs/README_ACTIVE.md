# 当前维护的多模态评估入口

当前可复现主线为 expanded_v3。此页列出当前维护入口；历史实验说明见 README_EXPERIMENTS.md。

- scripts/benchmarks/benchmark_multimodal_resources_unified.py：统一资源基准。
- scripts/evaluations/evaluate_*.py：E0–E4 模型准确率评估。
- scripts/datasets/build_node_text_profiles.py 和 build_node_text_profiles_en.py：节点文本画像。
- scripts/datasets/freeze_expanded_v3_images.py：冻结原始图像及校验和。
- scripts/reporting/restructure_comparison_tables.py：生成 A-1 至 C-2 对比表。
- scripts/reporting/append_resource_tables.py：追加资源表 D/E。

旧版 Graphify、通用测试集和备份不属于此维护目录。
