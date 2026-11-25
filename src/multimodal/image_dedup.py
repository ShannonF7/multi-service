"""兼容旧命令入口。

输入：原有 CLI 参数及可选模块属性访问。
输出：转发到 scripts 子包后的相同执行结果。
逻辑：延迟导入新实现，保持旧的 python -m 路径可用。
"""

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPLEMENTATION_MODULE: str = "src.multimodal.scripts.audits.image_dedup"


def _get_implementation() -> ModuleType:
    """导入新脚本模块并返回其实现对象。"""
    return import_module(_IMPLEMENTATION_MODULE)


def __getattr__(name: str) -> Any:
    """输入旧模块属性名，输出新实现中的对应属性。"""
    return getattr(_get_implementation(), name)


def main() -> None:
    """输入原命令行参数，输出由新实现产生的 CLI 结果。"""
    _get_implementation().main()


if __name__ == "__main__":
    main()
