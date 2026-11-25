"""PublishedEntity 的身份/内容角色判定。

角色是实体解析边界，不是某个具体 Intent 的白名单。
旧投影没有 entity_role 时，使用可解释的兼容推断；显式角色始终优先。
"""

from __future__ import annotations

import re
from typing import Any, Literal

EntityRole = Literal["IDENTITY", "CONTENT", "UNKNOWN"]

IDENTITY_NODE_TYPES = {
    "scenicarea", "scenic_area", "poi", "building", "room", "floor",
    "facility", "organization", "department", "person", "place",
    "location", "campus", "region", "area", "office", "laboratory",
    "门", "楼", "房间", "机构", "部门", "人物", "地点", "设施",
}
CONTENT_NODE_TYPES = {
    "article", "guide", "news", "introduction", "material", "content",
    "document", "text", "post", "资讯", "文章", "指南", "材料", "内容",
}
CONTENT_PARENT_NAMES = {"其他", "文章", "资料", "内容", "资讯", "材料"}
CONTENT_TITLE_MARKERS = (
    "指南", "介绍", "新闻", "电影电视剧", "非物质文化遗产", "生态保护",
    "四季游", "交通指南", "停车场指南", "游览指南", "导游服务", "预约指南",
    "摄影指南", "教育项目", "文创", "文化表演", "旅行健康", "药品与工具",
    "游客接待量", "投诉指南", "天气应对", "游客", "紧急情况", "购买",
    "探索之旅", "活动时间表", "营业时间及购票", "导览图", "全面覆盖",
    "体验服务", "提示", "政策", "禁烟防火", "儿童探索", "文化传承",
)


def normalize_role(value: Any) -> EntityRole | None:
    value = str(value or "").strip().upper()
    if value in {"IDENTITY", "IDENTITY_ENTITY"}:
        return "IDENTITY"
    if value in {"CONTENT", "CONTENT_ENTITY"}:
        return "CONTENT"
    if value == "UNKNOWN":
        return "UNKNOWN"
    return None


def infer_entity_role(item: dict[str, Any]) -> EntityRole:
    """根据显式角色、节点类型、父级和名称推断候选角色。"""
    explicit = normalize_role(item.get("entity_role"))
    if explicit is not None:
        return explicit

    node_type = _norm(item.get("node_type"))
    if node_type in CONTENT_NODE_TYPES:
        return "CONTENT"
    if node_type in IDENTITY_NODE_TYPES:
        return "IDENTITY"

    match_type = _norm(item.get("match_type"))
    if match_type in {"semantic_description", "semantic_evidence"}:
        return "CONTENT"

    parent_name = _norm(item.get("parent_name"))
    if parent_name in CONTENT_PARENT_NAMES:
        return "CONTENT"

    name = str(item.get("name") or "").strip()
    if any(marker in name for marker in CONTENT_TITLE_MARKERS):
        return "CONTENT"

    # 有坐标，或处在明确的层级父节点下，通常是可定位的领域实体。
    if item.get("lng") not in (None, "") or item.get("lat") not in (None, ""):
        return "IDENTITY"
    if parent_name:
        return "IDENTITY"
    return "UNKNOWN"


def _norm(value: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "").strip().casefold())
