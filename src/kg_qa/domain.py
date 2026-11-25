"""领域上下文和硬隔离入口。"""

from __future__ import annotations

from typing import Callable, Optional, Set

from .schemas import DomainContext, QARequest


class DomainAccessError(ValueError):
    """领域不存在、无权限或范围参数不合法。"""


class DomainGuard:
    """在进入任何检索器前建立并校验 DomainContext。

    resolver 返回图谱实际领域标识；source_scenic_id 默认沿用对外领域标识，
    也可以由调用方提供映射函数，处理 scenic code 与内部 ID 不一致的情况。
    """

    def __init__(
        self,
        resolver: Optional[Callable[[str], str]] = None,
        source_id_resolver: Optional[Callable[[str], str]] = None,
        allowed_domains: Optional[Set[str]] = None,
    ) -> None:
        self._resolver = resolver or self._default_graph_domain_resolver
        self._source_id_resolver = source_id_resolver or (lambda value: value)
        self._allowed_domains = allowed_domains

    @staticmethod
    def _default_graph_domain_resolver(domain_id: str) -> str:
        from src.rag.service.graph_discovery_service import resolve_published_domain_id

        return str(resolve_published_domain_id(domain_id) or "").strip()

    def validate(self, request: QARequest) -> DomainContext:
        domain_id = str(request.domain_id or "").strip()
        if not domain_id:
            raise DomainAccessError("domain_id is required")
        if self._allowed_domains is not None and domain_id not in self._allowed_domains:
            raise DomainAccessError("domain is not allowed")
        if request.scope in {"NODE", "SUBGRAPH"} and not str(request.current_node_id or "").strip():
            raise DomainAccessError("current_node_id is required for NODE or SUBGRAPH scope")

        graph_domain_id = str(self._resolver(domain_id) or "").strip()
        if not graph_domain_id:
            raise DomainAccessError("published graph domain not found")
        source_scenic_id = str(self._source_id_resolver(domain_id) or "").strip()
        if not source_scenic_id:
            raise DomainAccessError("source scenic id not found")
        return DomainContext(
            domain_id=domain_id,
            source_scenic_id=source_scenic_id,
            graph_domain_id=graph_domain_id,
            name=None,
            scope=request.scope,
            current_node_id=str(request.current_node_id or "").strip() or None,
        )
