"""本地决策模型运行前检查。

只检查文件和已安装运行时，不加载模型权重、不发起网络请求、不占用 GPU 推理资源。
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class LocalRuntimeStatus:
    model_path: Optional[str]
    model_directory_exists: bool
    config_exists: bool
    weight_files: List[str] = field(default_factory=list)
    transformers_available: bool = False
    torch_available: bool = False
    cuda_devices: List[str] = field(default_factory=list)
    ready: bool = False
    reasons: List[str] = field(default_factory=list)


def inspect_transformers_runtime(model_path: Optional[str] = None) -> LocalRuntimeStatus:
    """检查进程内 Transformers 后端是否具备加载条件。"""
    selected_path = model_path or os.getenv("LOCAL_DECISION_MODEL_PATH")
    path = Path(selected_path) if selected_path else None
    directory_exists = bool(path and path.is_dir())
    config_exists = bool(path and (path / "config.json").is_file())
    weight_files = _find_weight_files(path) if directory_exists else []
    transformers_available = importlib.util.find_spec("transformers") is not None
    torch_available = importlib.util.find_spec("torch") is not None
    cuda_devices = _cuda_devices() if torch_available else []

    reasons: list[str] = []
    if not selected_path:
        reasons.append("未配置 LOCAL_DECISION_MODEL_PATH")
    elif not directory_exists:
        reasons.append("模型目录不存在")
    if directory_exists and not config_exists:
        reasons.append("模型目录缺少 config.json")
    if directory_exists and not weight_files:
        reasons.append("模型目录未发现 safetensors 或 pytorch 权重文件")
    if not transformers_available:
        reasons.append("未安装 transformers")
    if not torch_available:
        reasons.append("未安装 torch")

    return LocalRuntimeStatus(
        model_path=str(path) if path else None,
        model_directory_exists=directory_exists,
        config_exists=config_exists,
        weight_files=weight_files,
        transformers_available=transformers_available,
        torch_available=torch_available,
        cuda_devices=cuda_devices,
        ready=not reasons,
        reasons=reasons,
    )


def _find_weight_files(path: Optional[Path]) -> list[str]:
    if path is None:
        return []
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    values = {item.name for pattern in patterns for item in path.glob(pattern) if item.is_file()}
    return sorted(values)


def _cuda_devices() -> list[str]:
    try:
        import torch

        return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []
