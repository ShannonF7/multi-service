"""问题请求目录：把自然语言表达映射为可审计的 QueryRequest 语义。

这里集中维护的是语义别名和轻量规则，不包含具体实体名称，也不判断图谱中是否存在答案。
是否存在信息必须由已发布图谱、属性、关系和图片检索结果决定。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class RequestRule:
    request_type: str
    markers: tuple[str, ...]
    attribute: str | None = None
    requires_target: bool = False


REQUEST_RULES: tuple[RequestRule, ...] = (
    RequestRule("NEARBY", ("附近有什么", "附近有", "周边有什么", "周围有什么", "附近有哪些", "周边有哪些", "周围有哪些", "附近"), requires_target=True),
    RequestRule("VISUAL_APPEARANCE", ("长什么样", "长啥样", "什么样子", "外观", "样貌", "现场图"), requires_target=True),
    RequestRule("IMAGE_RETRIEVAL", ("给我看看", "图片", "照片", "图像", "配图"), attribute="image", requires_target=True),
    RequestRule("LOCATION", ("在哪里", "在哪", "位于", "位置", "地址"), attribute="location"),
)

ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "manager": ("管理员", "负责人", "管理人", "管理者", "manager"),
    "contact": ("联系人", "联系方式", "联系信息", "contact"),
    "phone": ("电话", "手机", "联系电话", "电话号码", "phone", "telephone"),
    "email": ("邮箱", "电子邮箱", "电子邮件", "邮件地址", "email", "e-mail"),
    "area": ("面积", "占地面积", "建筑面积", "多大", "多少平方米", "area"),
    "name": ("名称", "名字", "名为", "name"),
}

ATTRIBUTE_DISPLAY: dict[str, str] = {
    "manager": "管理员",
    "contact": "联系方式",
    "phone": "电话",
    "email": "邮箱",
    "area": "面积",
    "name": "名称",
}


def find_rule_markers(text: str, request_type: str | None = None) -> list[tuple[int, RequestRule, str]]:
    value = str(text or "")
    wanted = str(request_type or "").upper()
    found: list[tuple[int, RequestRule, str]] = []
    for rule in REQUEST_RULES:
        if wanted and rule.request_type != wanted:
            continue
        for marker in rule.markers:
            position = value.find(marker)
            if position >= 0:
                found.append((position, rule, marker))
    return sorted(found, key=lambda item: (item[0], -len(item[2])))


def markers_for(request_type: str) -> tuple[str, ...]:
    wanted = str(request_type or "").upper()
    return tuple(marker for rule in REQUEST_RULES if rule.request_type == wanted for marker in rule.markers)


def attribute_for_text(text: str) -> str | None:
    value = str(text or "").casefold()
    matches: list[tuple[int, int, str]] = []
    for attribute, aliases in ATTRIBUTE_ALIASES.items():
        for alias in aliases:
            position = value.find(alias.casefold())
            if position >= 0:
                matches.append((position, -len(alias), attribute))
    return sorted(matches)[0][2] if matches else None


def attribute_aliases(attribute: str | None) -> tuple[str, ...]:
    return ATTRIBUTE_ALIASES.get(str(attribute or "").strip().casefold(), ())


def attribute_display(attribute: str | None) -> str:
    key = str(attribute or "").strip().casefold()
    return ATTRIBUTE_DISPLAY.get(key, key or "相关信息")


def graph_predicates(attributes: Iterable[str | None]) -> list[str]:
    values: list[str] = []
    for attribute in attributes:
        aliases = attribute_aliases(attribute)
        if aliases:
            values.extend(aliases)
        elif attribute:
            values.append(str(attribute).strip())
    return list(dict.fromkeys(value for value in values if value))


def fast_request_coverage(text: str) -> set[str]:
    """返回规则能直接识别出的请求类型，用来决定是否安全跳过轻量模型。"""
    return {rule.request_type for _, rule, _ in find_rule_markers(text)}


def can_fast_parse(text: str) -> bool:
    """仅在每个问句片段都能解析时跳过轻量结构化模型。"""
    segments = [item.strip() for item in re.split(r"[？?!！。；;]+", str(text or "")) if item.strip()]
    return bool(segments) and all(
        fast_request_coverage(segment) or attribute_for_text(segment)
        for segment in segments
    )

