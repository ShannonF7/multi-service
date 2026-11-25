"""第一阶段的问题理解器。

采用可审计的规则完成意图与问题要素识别，后续可以替换或叠加模型，
但输出必须继续符合 logical_form 中的数据契约。
"""

import re
from typing import List, Optional, Tuple

from .context import ConversationContextView, is_current_location_question, resolve_context
from .logical_form import Intent, LogicalQuery, ReasoningMode, TargetReference


_INTENT_PATTERNS: Tuple[Tuple[Intent, Tuple[str, ...]], ...] = (
    ("PATH", ("怎么走", "如何到", "路线", "路径", "导航", "步行", "驾车")),
    ("AGGREGATION", ("有哪些", "有几个", "多少个", "列出", "分别是")),
    ("COMPARISON", ("区别", "差异", "比较", "哪个更", "对比")),
    ("TEMPORAL", ("什么时候", "哪一年", "何时", "历史", "成立时间", "创建时间")),
    ("IMAGE", ("图片", "照片", "外观", "长什么样", "看一下")),
    ("HIERARCHY", ("属于", "下级", "上级", "包含", "组成", "位于哪一层", "隶属")),
    ("LOCATION", ("在哪里", "在哪儿", "哪儿", "位置", "地址", "坐标", "怎么去", "位于")),
    ("EXPLANATION", ("介绍", "是什么", "什么是", "为什么", "简介", "说明")),
)


def _intent_for(question: str) -> Tuple[Intent, float, str]:
    text = question or ""
    if is_current_location_question(text):
        return "CURRENT_LOCATION", 0.99, "current_location_rule"
    for intent, markers in _INTENT_PATTERNS:
        if any(marker in text for marker in markers):
            return intent, 0.86, "keyword_rule"
    if re.search(r"(吗|么|？|\?)$", text.strip()):
        return "FACT", 0.55, "question_fallback"
    return "CHAT", 0.90, "casual_fallback"


def _extract_mentions(question: str, intent: Intent) -> List[str]:
    """提取带层级限定词的完整短语，不把上层限定词截掉。"""

    text = re.sub(r"\s+", "", question or "")
    if not text or intent == "CURRENT_LOCATION":
        return []
    quoted = re.findall(r"[“「『]([^”」』]+)[”」』]", text)
    if quoted:
        return [item.strip() for item in quoted if item.strip()]

    match = re.search(r"(.+?)(在哪里|在哪儿|哪儿|位置|地址|坐标|怎么去|位于|是什么|什么是|有哪些|有几个|多少个|介绍|图片|照片|外观)", text)
    if match:
        value = match.group(1).strip("，,：:？?。 ")
    else:
        value = re.sub(r"^(请问|请|帮我|你能否|能否|告诉我)", "", text).strip("，,：:？?。 ")
    value = re.sub(r"^(关于|有关)", "", value).strip()
    value = re.sub(r"^(请介绍一下|介绍一下|介绍)", "", value).strip()
    value = value.strip("“”‘’")
    if not value or value in {"我", "现在", "当前", "这里", "那里"}:
        return []
    return [value]


def _relation_for(intent: Intent) -> Optional[str]:
    return {
        "LOCATION": "located_in",
        "CURRENT_LOCATION": "located_in",
        "HIERARCHY": "parent_of",
        "PATH": "route",
        "TEMPORAL": "has_time",
    }.get(intent)


def _mode_for(intent: Intent) -> ReasoningMode:
    return {
        "CHAT": "NONE",
        "CURRENT_LOCATION": "CURRENT_LOCATION",
        "LOCATION": "DIRECT_RELATION",
        "HIERARCHY": "HIERARCHY",
        "PATH": "PATH_SEARCH",
        "AGGREGATION": "AGGREGATION",
        "COMPARISON": "COMPARISON",
        "TEMPORAL": "TEMPORAL",
        "IMAGE": "IMAGE_EVIDENCE",
        "EXPLANATION": "MULTI_HOP",
        "RELATION": "MULTI_HOP",
    }.get(intent, "DIRECT_PROPERTY")


def understand_question(
    question: str,
    domain_id: str,
    context: Optional[ConversationContextView] = None,
    target_node_id: Optional[str] = None,
    target_name: Optional[str] = None,
) -> LogicalQuery:
    """把自然语言问题编译成逻辑查询，不访问图谱。"""

    resolved = resolve_context(question, context)
    intent, confidence, _ = _intent_for(resolved.standalone_question)
    targets: List[TargetReference] = []

    if resolved.active_entity_id or resolved.active_entity_name:
        targets.append(
            TargetReference(
                mention=resolved.active_entity_name or str(resolved.active_entity_id),
                node_id=resolved.active_entity_id,
                canonical_name=resolved.active_entity_name,
                source=resolved.active_entity_source or "CONTEXT",
            )
        )

    if intent == "CHAT":
        targets = []

    if intent != "CHAT" and (target_node_id is not None or target_name):
        targets = [
            TargetReference(
                mention=target_name or str(target_node_id),
                node_id=target_node_id,
                canonical_name=target_name,
                source="EXPLICIT",
            )
        ]
    elif intent != "CHAT" and not targets:
        mentions = _extract_mentions(resolved.standalone_question, intent)
        targets = [TargetReference(mention=item) for item in mentions]

    if intent == "CURRENT_LOCATION" and not (resolved.active_entity_id or resolved.active_entity_name):
        targets = []

    constraints = {}
    if resolved.pending_selection:
        constraints["selected_candidate_index"] = resolved.pending_selection.index
    if intent in {"PATH", "AGGREGATION", "COMPARISON"}:
        constraints["bounded"] = True

    return LogicalQuery(
        question=resolved.original_question,
        standalone_question=resolved.standalone_question,
        domain_id=str(domain_id).strip(),
        intent=intent,
        relation=_relation_for(intent),
        targets=tuple(targets),
        constraints=constraints,
        reasoning_mode=_mode_for(intent),
        confidence=confidence,
        used_context=resolved.used_context,
        used_location=resolved.used_location,
    )
