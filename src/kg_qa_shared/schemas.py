"""KG-QA V1 与 V2 共用的请求数据契约。"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class UserLocation(BaseModel):
    """用户提交的位置坐标和采集信息。

    输入：经纬度，以及可选的位置精度和采集时间。
    输出：通过 Pydantic 范围校验的位置对象。
    逻辑：限制经纬度和精度取值，供 V1、V2 查询入口共同使用。
    """

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy_m: Optional[float] = Field(default=None, ge=0)
    captured_at_ms: Optional[int] = Field(default=None, ge=0)


class QARequest(BaseModel):
    """V1、V2 共用的在线问答请求。

    输入：领域、问题、会话、位置和多模态查询选项。
    输出：通过 Pydantic 校验的统一问答请求对象。
    逻辑：保持现有接口字段和约束，让两版服务消费同一份请求契约。
    """

    domain_id: str = Field(..., min_length=1)
    message: str = ""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    user_location: Optional[UserLocation] = None
    conversation_id: Optional[str] = None
    scope: Literal["NODE", "SUBGRAPH", "DOMAIN"] = "DOMAIN"
    current_node_id: Optional[str] = None
    image_ids: List[str] = Field(default_factory=list)
    image_path: Optional[str] = None
    target_modality: Optional[Literal["TEXT", "IMAGE"]] = None
    max_evidence: int = Field(default=8, ge=1, le=20)
