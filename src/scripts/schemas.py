from typing import Optional, Generic, TypeVar, List
from pydantic import BaseModel
from enum import Enum

# Error Codes
class ErrorCode(int, Enum):
    SUCCESS = 0
    PARAM_ERROR = 40001
    BUSINESS_ERROR = 40002
    UNAUTHORIZED = 40100
    FORBIDDEN = 40300
    NOT_FOUND = 40400
    INTERNAL_ERROR = 50000

T = TypeVar("T")

class BaseResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "ok"
    data: Optional[T] = None

class PaginatedData(BaseModel, Generic[T]):
    items: List[T]
    page: int
    page_size: int
    total: int
    total_pages: int

class GenerateScriptRequest(BaseModel):
    team_id: str
    style: str
    template_id: str

class ChatRequest(BaseModel):
    team_id: str
    user_id: str
    task_id: str
    message: str
    session_id: str
    history: Optional[list] = []
    generated_script_id: str
    task_status: Optional[str] = "in_progress"  # "in_progress" or "completed"
    sub_task_id: Optional[str] = None
    image_result: Optional[dict] = None # { "success": true, "message": "..." }
