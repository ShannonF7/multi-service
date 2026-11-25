"""现有 KG-QA 服务与新推理核心之间的适配层。

适配层只负责把旧 session/domain 对象转换为新推理层需要的输入，
不执行数据库操作，也不改变旧服务对象。
"""

from typing import Any, Optional

from src.kg_qa.schemas import DomainContext, QARequest

from .context import ConversationContextView
from .logical_form import LogicalQuery
from .query_understanding import understand_question


def build_context_view(
    session_context: Any,
    domain_context: DomainContext,
    request: QARequest,
) -> ConversationContextView:
    """合并 session 与当前请求的位置，生成推理层会话视图。"""

    if hasattr(session_context, "to_dict"):
        value = dict(session_context.to_dict())
    elif isinstance(session_context, dict):
        value = dict(session_context)
    else:
        value = {}
    value["location_node_id"] = (
        domain_context.location_node_id
        or request.current_node_id
        or value.get("location_node_id")
    )
    value["location_node_name"] = (
        domain_context.location_node_name
        or value.get("location_node_name")
    )
    return ConversationContextView.from_mapping(value)


def build_logical_query(
    message: str,
    domain_context: DomainContext,
    session_context: Any,
    request: QARequest,
    *,
    target_node_id: Optional[str] = None,
    target_name: Optional[str] = None,
) -> LogicalQuery:
    """将旧服务输入编译为新推理逻辑查询。"""

    view = build_context_view(session_context, domain_context, request)
    return understand_question(
        message,
        domain_context.graph_domain_id,
        context=view,
        target_node_id=target_node_id,
        target_name=target_name,
    )
