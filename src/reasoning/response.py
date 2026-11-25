"""面向用户的问答状态文案。

本模块只负责把内部状态转换成自然、可继续对话的中文，
不改变检索结果，也不隐藏真实的不确定性。
"""

from typing import Iterable


def unresolved_position_answer(names: Iterable[str]) -> str:
    """生成未找到位置时的自然回复，不暴露内部检索实现。"""

    values = [str(name).strip() for name in names if str(name).strip()]
    label = "、".join(values) or "这个位置"
    return (
        f"我暂时还没在当前领域的已发布信息中找到“{label}”。"
        "你可以补充一下所在校区、楼栋或部门，我再帮你缩小范围；"
        "也可以换一种名称试试。"
    )
