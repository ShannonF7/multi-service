"""旧 api.py 导入路径的兼容转发。"""

from .router import (
    _resolve_location,
    _service_kwargs,
    query,
    router,
    service,
    stream,
    upload_image,
)
from .schemas import QAResponse, UploadImageData

__all__ = [
    "router",
    "service",
    "query",
    "stream",
    "upload_image",
    "_resolve_location",
    "_service_kwargs",
    "QAResponse",
    "UploadImageData",
]
