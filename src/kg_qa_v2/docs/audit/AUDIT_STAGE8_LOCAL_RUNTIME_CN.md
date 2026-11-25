# 阶段 8 本地运行前检查审计

## 输入

输入是模型目录路径，默认读取 `LOCAL_DECISION_MODEL_PATH`。检查器只读取目录结构和 Python 模块是否可发现，不读取权重内容。

## 输出

输出 `LocalRuntimeStatus`，包含目录、配置文件、权重文件、Transformers、Torch、CUDA 设备和 `ready/reasons`。`ready=True` 只表示具备进入加载阶段的基本条件，不代表模型已经成功加载或推理质量合格。

## 安全边界

本阶段没有模型加载、没有网络下载、没有 GPU 推理、没有环境安装。权重文件只识别 `safetensors/bin/pt/pth` 文件名，不把缓存引用目录误判成可运行模型。

## 测试结果

覆盖缺失目录、缺少权重、最小模型目录三种情况。与前面阶段合计 31 个测试通过；`compileall` 通过。
