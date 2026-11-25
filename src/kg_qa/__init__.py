"""在线 KG-QA 模块。

该包只负责面向用户的问答编排，不参与语义补全候选写入或自增长发布。
"""

from .schemas import QARequest, QAResponse

__all__ = ["QARequest", "QAResponse"]
