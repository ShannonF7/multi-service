import logging
import os
import time
import datetime
import json
import importlib.util
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import iterate_in_threadpool
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from src.scripts.router import router as script_router
from src.llm.router import router as llm_router
from src.database.session import get_db
from src.scripts.schemas import BaseResponse, ErrorCode
from src.core.security import SecurityMiddleware
from src.cv.feature_extractor import get_feature_extractor
from src.rag.sync_router import router as rag_sync_router
from src.rag.semantic_router import router as rag_semantic_router
from src.rag.graph_router import router as rag_graph_router
from src.semantic_growth.api import router as semantic_growth_router

# Logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Travel API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Feature Extractor Model...")
    # Initialize the singleton
    get_feature_extractor()
    logger.info("Feature Extractor Model Initialized.")

# 1. Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Map HTTP status codes to custom ErrorCode if possible, or use generic logic
    custom_code = exc.status_code
    if exc.status_code == 400:
        custom_code = ErrorCode.PARAM_ERROR.value
    elif exc.status_code == 401:
        custom_code = ErrorCode.UNAUTHORIZED.value
    elif exc.status_code == 403:
        custom_code = ErrorCode.FORBIDDEN.value
    elif exc.status_code == 404:
        custom_code = ErrorCode.NOT_FOUND.value
    elif exc.status_code == 500:
        custom_code = ErrorCode.INTERNAL_ERROR.value
    
    
    return JSONResponse(
        status_code=exc.status_code, # HTTP status code remains standard
        content=BaseResponse(
            code=custom_code,
            message=exc.detail,
            data=None
        ).dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=BaseResponse(
            code=ErrorCode.INTERNAL_ERROR.value,
            message="Internal Server Error",
            data=str(exc) if os.getenv("DEBUG") else None
        ).dict()
    )

# 2. Security Middleware (Imported from src.core.security)
# app.add_middleware(SecurityMiddleware) is called below

# 3. Logging Middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Define paths to ignore detailed logging for
        ignored_paths = ["/docs", "/redoc", "/openapi.json", "/favicon.ico"]
        should_log_details = (
            request.url.path not in ignored_paths
            and request.url.path != "/rag/graph/sync/jobs"
        )

        # Log Request
        request_body_content = ""
        if should_log_details:
            # 对 SSE / 流式端点，不要调用 request.body()（会与 BaseHTTPMiddleware 的 receive 管道冲突）
            # 只记录一个占位符，保留原有日志格式。
            if (
                request.url.path.endswith("/chat/stream") or 
                request.url.path.endswith("/chat/stream/") or
                request.url.path.endswith("/task/evaluation/stream") or 
                request.url.path.endswith("/task/evaluation/stream/") or
                request.url.path.endswith("/chat/test/") or
                request.url.path.endswith("/chat/test") or
                request.url.path.endswith("/npc/chat/stream_with_prompt/") or
                request.url.path.endswith("/npc/chat/stream_with_prompt")
            ):
                request_body_content = "<Streaming Request Body Skipped>"
            else:
                content_type = request.headers.get("content-type", "")

                if "multipart/form-data" in content_type:
                    request_body_content = "[Multipart/Form-Data File Upload]"
                else:
                    try:
                        # Read request body
                        body_bytes = await request.body()
                        # Reset body so it can be read again by the app
                        async def receive():
                            return {"type": "http.request", "body": body_bytes, "more_body": False}
                        request._receive = receive

                        if body_bytes:
                            request_body_content = body_bytes.decode()
                            if "application/json" in content_type:
                                try:
                                    data = json.loads(request_body_content)
                                    if isinstance(data, dict) and "history" in data:
                                        data.pop("history")
                                        request_body_content = json.dumps(data, ensure_ascii=False)
                                except:
                                    pass
                    except Exception:
                        request_body_content = "<Binary or Stream>"

            logger.info(f"[Request] {request.method} {request.url.path} - 请求数据: {request_body_content}")
        else:
            logger.info(f"[Request] {request.method} {request.url.path} - (Log Skipped for Doc/Static)")

        try:
            response = await call_next(request)
        except Exception as e:
            raise e

        process_time = (time.time() - start_time) * 1000
        formatted_process_time = "{0:.2f}".format(process_time)
        
        response_body_content = ""
        if should_log_details:
            content_type = (response.headers.get("content-type", "") or "").lower()
            if "text/event-stream" in content_type:
                logger.info(
                    f"[Response] {request.method} {request.url.path} - 响应数据: <StreamingResponse text/event-stream> - 耗时: {formatted_process_time} ms"
                )
                return response
            try:
                # Consuming response body to log it
                response_body = [section async for section in response.body_iterator]
                response.body_iterator = iterate_in_threadpool(iter(response_body))
                if response_body:
                    response_body_content = b"".join(response_body).decode()
            except Exception:
                response_body_content = "<Binary or Stream>"

            logger.info(f"[Response] {request.method} {request.url.path} - 响应数据: {response_body_content} - 耗时: {formatted_process_time} ms")
        else:
            logger.info(f"[Response] {request.method} {request.url.path} - (Log Skipped) - 耗时: {formatted_process_time} ms")
        
        return response

app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(script_router, prefix="/api/v1", tags=["Script Generation & Chat"])
app.include_router(llm_router, prefix="/api/v1", tags=["Task Judge & Advance"])

# A端“一键入库”写入的是 AI_DB 中的 RAG 结构化表，不使用 optimized 主业务库。
# 这里保持根路径 /sync/*，让 A 端只需要配置 B 端服务根地址即可提交。
# 接口仍经过全局 SecurityMiddleware；生产环境需携带 X-API-KEY。
app.include_router(rag_sync_router, tags=["A端一键入库 / RAG Sync"])
app.include_router(rag_semantic_router, tags=["RAG Semantic Completion"])
app.include_router(rag_graph_router, tags=["Published Graph Projection"])
app.include_router(semantic_growth_router)


def _mount_420pro_app() -> None:
    module_path = Path(__file__).resolve().parent / "src" / "images" / "420pro" / "main_api.py"
    if not module_path.exists():
        logger.warning("420pro main_api.py not found, skip mounting")
        return

    spec = importlib.util.spec_from_file_location("src.images.420pro.main_api", module_path)
    if spec is None or spec.loader is None:
        logger.warning("Unable to load 420pro app spec, skip mounting")
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sub_app = getattr(module, "app", None)
    if sub_app is None:
        logger.warning("420pro app instance not found, skip mounting")
        return

    app.mount("/api/v1/420pro", sub_app)


_mount_420pro_app()

# Use absolute path for templates to avoid CWD issues
base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))


@app.get("/db")
def list_tables(request: Request, db: Session = Depends(get_db)):
    inspector = inspect(db.bind)
    tables = inspector.get_table_names()

    return templates.TemplateResponse(
        "table_list.html",
        {
            "request": request,
            "tables": tables
        }
    )


@app.get("/db/{table_name}")
def view_table(
    table_name: str,
    request: Request,
    db: Session = Depends(get_db)
):
    inspector = inspect(db.bind)

    if table_name not in inspector.get_table_names(schema='public'):
        raise HTTPException(status_code=404, detail="表不存在（可能不在public schema里哦）")

    result = db.execute(text(f"SELECT * FROM public.{table_name} LIMIT 50"))
    rows = result.fetchall()
    columns = result.keys()

    return templates.TemplateResponse(
        "table_detail.html",
        {
            "request": request,
            "table_name": table_name,
            "columns": columns,
            "rows": rows
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
