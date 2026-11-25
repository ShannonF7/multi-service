"""KG-QA V2 HTTP 请求与响应的数据结构。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.kg_qa_shared.schemas import QARequest, UserLocation


class QAResponse(BaseModel):
    """KG-QA V2 查询结果的数据模型。

    输入：问答服务生成的答案、引用、证据和运行信息。
    输出：供统一响应 data 字段序列化的结构化结果。
    逻辑：保留 V2 查询结果字段及其默认值。
    """

    status: str
    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    process: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: int = 0
    timings: dict[str, int] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    route: Optional[str] = None
    intent: Optional[str] = None
    clarification: Optional[dict[str, Any]] = None



class UploadImageData(BaseModel):
    """图片上传成功后放入统一响应 data 字段的内容。

    输入：已保存图片的路径、文件名、类型和大小。
    输出：由统一响应 data 字段承载的图片信息对象。
    逻辑：只描述上传结果，不承担文件校验或保存操作。
    """

    image_path: str
    filename: str
    content_type: str
    size: int


__all__ = ["QARequest", "UserLocation", "QAResponse", "UploadImageData"]
