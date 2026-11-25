"""KG-QA v2 HTTP 路由。

保留旧版 /kg-qa 路由作为回退；本模块提供 /kg-qa-v2，负责把 A 端请求
转换成 QAV2Service 的统一输入，并以标准 SSE 帧输出事件。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Iterator
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.scripts.schemas import BaseResponse
from .core.streaming import encode_sse
from .schemas import QARequest, QAResponse, UploadImageData, UserLocation
from .service import QAV2Service

router: APIRouter = APIRouter(prefix="/kg-qa-v2", tags=["KG-QA v2"])
service: QAV2Service = QAV2Service.from_env()


def _resolve_location(payload: QARequest) -> dict[str, Any] | None:
    """解析请求中的实时位置并附加当前领域的最近已发布节点。

    输入：包含 GPS 或当前节点 ID 的 QARequest。
    输出：位置字段字典；没有位置时返回 None。
    逻辑：先检查领域范围，再查找范围内最近的已发布节点；查询失败时保留坐标。
    """
    location: UserLocation | None = payload.user_location
    if location is None and payload.current_node_id:
        return {"node_id": str(payload.current_node_id)}
    if location is None:
        return None

    result: dict[str, Any] = {
        "latitude": location.latitude,
        "longitude": location.longitude,
    }
    if location.accuracy_m is not None:
        result["accuracy_m"] = location.accuracy_m
    if location.captured_at_ms is not None:
        result["captured_at_ms"] = location.captured_at_ms
    scope: dict[str, Any] | None = None
    try:
        from src.rag.service.graph_discovery_service import resolve_published_domain_scope
        scope = resolve_published_domain_scope(payload.domain_id)
    except Exception:
        scope = None
    in_scope: bool | None = None
    if scope:
        center: dict[str, Any] = scope.get("center") or {}
        try:
            dx: float = (
                (float(location.longitude) - float(center["longitude"]))
                * 111320
                * math.cos(math.radians(float(location.latitude)))
            )
            dy: float = (
                (float(location.latitude) - float(center["latitude"])) * 110540
            )
            distance_to_domain: float = round((dx * dx + dy * dy) ** 0.5, 2)
            radius_m: float = float(scope.get("radius_m") or 0)
            in_scope = distance_to_domain <= radius_m
            result.update({
                "domain_name": scope.get("domain_name"),
                "domain_center": center,
                "domain_scope_radius_m": radius_m,
                "distance_to_domain_m": distance_to_domain,
                "location_in_domain": in_scope,
            })
        except (KeyError, TypeError, ValueError):
            in_scope = None
    try:
        from src.rag.service.graph_discovery_service import resolve_nearest_published_node
        nearest: dict[str, Any] | None = (
            resolve_nearest_published_node(
                payload.domain_id,
                latitude=location.latitude,
                longitude=location.longitude,
            )
            if in_scope is not False
            else None
        )
    except Exception:
        nearest = None
    if nearest:
        result.update({
            "node_id": str(nearest.get("node_id") or "") or None,
            "name": str(nearest.get("name") or "") or None,
            "distance_m": nearest.get("distance_m"),
            "node_type": nearest.get("node_type"),
        })
    elif in_scope is False:
        result.update({"node_id": None, "name": None, "distance_m": None})
    return result


def _service_kwargs(payload: QARequest) -> dict[str, Any]:
    """把 API 请求转换为问答服务参数。

    输入：通过 Pydantic 校验的 QARequest。
    输出：QAV2Service.ask 和 stream_ask 共用的关键字参数字典。
    逻辑：映射领域、会话、图片和证据上限，并解析实时位置。
    """

    return {
        "question": payload.message,
        "domain_id": payload.domain_id,
        "session_id": payload.session_id or payload.conversation_id,
        "user_id": payload.user_id,
        "current_location": _resolve_location(payload),
        "has_image": bool(payload.image_path or payload.image_ids),
        "image_path": payload.image_path,
        "target_modality": payload.target_modality,
        "max_evidence": payload.max_evidence,
    }

_IMAGE_ALLOWED_ROOT: Path = Path(
    os.getenv(
        "QWEN_VL_EMBEDDING_ALLOWED_ROOT",
        str(Path(__file__).resolve().parents[2] / "data" / "domain_kb"),
    )
).resolve()
_CONFIGURED_UPLOAD_ROOT: Path = Path(
    os.getenv("KG_QA_UPLOAD_DIR", str(_IMAGE_ALLOWED_ROOT / ".qa_uploads"))
).resolve()
# 图片向量服务只允许读取允许目录；上传目录必须位于其下。
_UPLOAD_ROOT: Path = (
    _CONFIGURED_UPLOAD_ROOT
    if _CONFIGURED_UPLOAD_ROOT == _IMAGE_ALLOWED_ROOT
    or _IMAGE_ALLOWED_ROOT in _CONFIGURED_UPLOAD_ROOT.parents
    else _IMAGE_ALLOWED_ROOT / ".qa_uploads"
)
_MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024


@router.post("/upload", response_model=BaseResponse[UploadImageData])
def upload_image(file: UploadFile = File(...)) -> BaseResponse[UploadImageData]:
    """接收图片并返回统一格式的上传结果。

    输入：multipart/form-data 中名为 file 的图片文件。
    输出：BaseResponse[UploadImageData]，data 包含临时文件路径和元信息。
    逻辑：校验图片类型、后缀和大小，保存到允许目录后返回文件信息。
    """
    content_type: str = str(file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="只支持图片文件")
    filename: str = Path(str(file.filename or "upload")).name
    suffix: str = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        suffix = ".bin"
    content: bytes = file.file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 10 MB")
    if not content:
        raise HTTPException(status_code=400, detail="图片内容为空")
    _UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    target: Path = _UPLOAD_ROOT / f"{uuid4().hex}{suffix}"
    target.write_bytes(content)
    return BaseResponse[UploadImageData](
        message="图片上传成功",
        data=UploadImageData(
            image_path=str(target),
            filename=filename,
            content_type=content_type,
            size=len(content),
        ),
    )


@router.post("/query", response_model=BaseResponse[QAResponse])
def query(payload: QARequest) -> BaseResponse[QAResponse]:
    """执行一次非流式 KG-QA 查询并包装为统一响应。

    输入：通过 Pydantic 校验的 QARequest。
    输出：BaseResponse[QAResponse]，业务结果位于 data 字段。
    逻辑：映射服务参数，调用 V2 问答服务并返回标准成功码和说明。
    """

    result: QAResponse = service.ask(**_service_kwargs(payload))
    return BaseResponse[QAResponse](message="问答查询成功", data=result)


@router.post("/stream")
def stream(payload: QARequest) -> StreamingResponse:
    """以 SSE 推送问答阶段、答案片段和最终结果。

    输入：通过 Pydantic 校验的 QARequest。
    输出：text/event-stream；事件类型由 V2 流服务定义。
    逻辑：后台线程消费问答事件并放入队列，生成器逐帧编码；异常转为 SSE error 事件。
    """

    queue: Queue[dict[str, Any] | None] = Queue()

    def worker() -> None:
        try:
            event: dict[str, Any]
            for event in service.stream_ask(**_service_kwargs(payload)):
                queue.put(event)
        except Exception as exc:
            queue.put({
                "event": "error",
                "data": {"message": str(exc)[:500]},
            })
        finally:
            queue.put(None)

    Thread(target=worker, name="kg-qa-v2-stream", daemon=True).start()

    def generate() -> Iterator[str]:
        while True:
            event: dict[str, Any] | None = queue.get()
            if event is None:
                break
            yield encode_sse(event)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
