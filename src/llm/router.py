# LLM interface(answer、script)
from fastapi import APIRouter, FastAPI, HTTPException, Depends, Request
from fastapi.concurrency import run_in_threadpool
from starlette.concurrency import iterate_in_threadpool
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
import logging
import uuid
import json
from src.database.session import get_db
from src.database.redis import get_redis
from src.scripts.schemas import ChatRequest, BaseResponse, ErrorCode
from src.llm.service import handle_dialogue_request, handle_dialogue_stream, _simple_parse_json, call_api_stream_with_retry, handle_dialogue_stream_test
from src.llm.prompts.prompts import zhangbi_game_prompt_v2
from src.database.models import GeneratedScript
import time
from datetime import datetime
import pytz
import requests
from pydantic import BaseModel
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/task/evaluation/", response_model=BaseResponse)
async def chat_with_npc_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Unified endpoint for NPC dialogue with Evaluation Engine.
    1. Loads task config from DB.
    2. Runs Evaluation Pipeline (Intent -> Score -> Rule).
    3. If Rule triggers, return strictly.
    4. Else, fallback to standard LLM chat (reusing legacy logic).
    """
    try:
        # Redis Session Management
        redis_client = await get_redis()
        session_id = request.session_id or str(uuid.uuid4())
        redis_key = f"chat_session:{session_id}"
        
        # Load history from Redis if request history is empty/partial
        history = []
        if request.session_id:
            cached_history = await redis_client.get(redis_key)
            if cached_history:
                try:
                    history = json.loads(cached_history)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode history for session {session_id}")
                    history = []
        
        # Fallback/Merge: If Redis empty, use client history
        if not history and request.history:
            history = request.history
            
        # Update request object with full history for service processing
        # Note: handle_dialogue_request expects ChatRequest, so we modify it in place or re-construct
        # Ideally we pass history explicitly, but ChatRequest property is immutable? Pydantic V2 compat.
        request.history = history 

        # Running the service logic in threadpool
        result = await run_in_threadpool(
            handle_dialogue_request, 
            request, 
            db
        )
        
        # Update History with new turn
        # 1. User Message
        new_history_entry_user = {"role": "user", "content": request.message, "user_id": request.user_id}
        
        # 2. Assistant Message
        assistant_content = result["reply"]

        new_history_entry_npc = {"role": "assistant", "content": assistant_content}
        
        history.append(new_history_entry_user)
        history.append(new_history_entry_npc)
        
        # Keep only last 30 turns
        if len(history) > 60:
            history = history[-60:]
            
        # Save to Redis
        await redis_client.setex(redis_key, 86400, json.dumps(history))
        
        final_result = {
            "reply": result.get("reply"),
            "task_completed": result.get("task_completed"),
            "session_id": session_id
        }
        
        return BaseResponse(
            code=ErrorCode.SUCCESS.value,
            message="ok",
            data=final_result
        )
    except Exception as e:
        logger.exception("Chat failed in Evaluation Endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/npc/chat/stream/")
# async def chat_with_npc_stream_endpoint(
#     request: ChatRequest,
#     db: Session = Depends(get_db)
# ):
#     """
#     Streaming endpoint for NPC dialogue with Evaluation Engine.
#     """
#     # Manual logging of request body since middleware skips it for streaming endpoints
#     try:
#         # Pydantic models can be converted to dict
#         req_dict = request.dict()
#         if "history" in req_dict:
#             req_dict.pop("history")
#         req_json = json.dumps(req_dict, ensure_ascii=False)
#     except:
#         req_json = str(request)
#     logger.info(f"[Request] POST /api/v1/npc/chat/stream - 请求数据: {req_json}")

#     async def event_generator():
#         start_time = time.time()
#         try:
#             # Redis Session Management
#             redis_client = await get_redis()
#             session_id = request.session_id or str(uuid.uuid4())
#             redis_key = f"chat_session:{session_id}"
            
#             # Load history
#             history = []
#             if request.session_id:
#                 cached = await redis_client.get(redis_key)
#                 if cached:
#                     try:
#                         history = json.loads(cached)
#                     except:
#                         history = []
            
#             if not history and request.history:
#                 history = request.history

#             # Add user message to history immediately for context? 
#             # Ideally context excludes current message, but prompt builder handles it.
            
#             full_reply = ""
#             final_data = {}

#             # Call Streaming Service
#             async for event_type, data in handle_dialogue_stream(
#                 team_id=request.team_id,
#                 user_id=request.user_id,
#                 task_id=request.task_id,
#                 message=request.message,
#                 history=history,
#                 db=db,
#                 generated_script_id=request.generated_script_id,
#                 task_status=request.task_status,
#                 sub_task_id=request.sub_task_id,
#                 image_result=request.image_result
#             ):
#                 if event_type == "delta":
#                     full_reply += data
#                     yield f"data: {json.dumps({'delta': data}, ensure_ascii=False)}\n\n"
#                 elif event_type == "final":
#                     # Strip action/action_data from streaming final event
#                     data.pop("action", None)
#                     data.pop("action_data", None)
#                     final_data = data
                    
#                     # Construct standard response payload
#                     response_payload = {
#                         "code": ErrorCode.SUCCESS.value,
#                         "message": "ok",
#                         "data": {**data, "session_id": session_id}
#                     }
#                     yield f"data: {json.dumps(response_payload, ensure_ascii=False)}\n\n"

#             # Update History after stream completes
#             # 1. User Message
#             new_history_entry_user = {"role": "user", "content": request.message, "user_id": request.user_id}
#             # 2. Assistant Message
#             assistant_content = final_data.get("reply") or full_reply

#             if assistant_content:
#                 history.append(new_history_entry_user)
#                 history.append({"role": "assistant", "content": assistant_content})
                
#                 if len(history) > 60:
#                     history = history[-60:]
                
#                 await redis_client.setex(redis_key, 86400, json.dumps(history))

#             # Manual Logging of final response content
#             process_time = (time.time() - start_time) * 1000
#             formatted_process_time = "{0:.2f}".format(process_time)
            
#             # Reconstruct response payload for logging (in case final event logic is separate)
#             log_payload = {
#                 "code": ErrorCode.SUCCESS.value,
#                 "message": "ok",
#                 "data": {**final_data, "session_id": session_id}
#             }
#             logger.info(
#                 f"[Response] POST /api/v1/npc/chat/stream - 响应数据: {json.dumps(log_payload, ensure_ascii=False)} - 耗时: {formatted_process_time} ms"
#             )

#             yield "data: [DONE]\n\n"

#         except Exception as e:
#             logger.exception("Stream failed")
#             err_payload = {
#                 "code": ErrorCode.INTERNAL_ERROR.value,
#                 "message": f"Stream failed: {str(e)}",
#                 "data": None
#             }
#             yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

#     return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/npc/chat/stream/")
async def chat_with_npc_stream_endpoint_test(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Streaming endpoint for NPC dialogue - Direct Implementation with RAG & Time Context
    """
    # Safe logging
    try:
        req_info = request.dict(exclude={"history"})
    except:
        req_info = {"request": "parsing_failed"}
    logger.info(f"[Request] POST /api/v1/npc/chat/test/ - {req_info}")

    async def event_generator():
        start_time = time.time()
        try:
            # 1. Init Session & Redis
            redis_client = await get_redis()
            session_id = request.session_id or str(uuid.uuid4())
            redis_key = f"chat_session:{session_id}"
            
            # 2. Context: Load history
            history = []
            if request.session_id:
                cached = await redis_client.get(redis_key)
                if cached:
                    try:
                        history = json.loads(cached)
                    except:
                        history = []
            
            if not history and request.history:
                history = request.history

            full_reply = ""
            final_data = {}

            # Prepare Context from RAG and Time Helper
            rag_context = "无"
            try:
                # Run sync RAG fetch in threadpool to avoid blocking event loop
                rag_context = await run_in_threadpool(fetch_rag_context, request.message)
            except Exception as e:
                logger.warning(f"RAG fetch failed: {e}")
            
            time_info = get_current_time_info()

            # 3. Call Service (handle_dialogue_stream_test)
            # This service function now contains the correct logic for sub-task descriptions and new fields
            async for event_type, data in handle_dialogue_stream_test(
                team_id=request.team_id,
                user_id=request.user_id,
                task_id=request.task_id,
                message=request.message,
                history=history,
                db=db,
                generated_script_id=request.generated_script_id,
                task_status=request.task_status,
                sub_task_id=request.sub_task_id,
                image_result=request.image_result,
                rag_context=rag_context,
                time_info=time_info
            ):
                if event_type == "delta":
                    full_reply += data
                    yield f"data: {json.dumps({'delta': data}, ensure_ascii=False)}\n\n"
                elif event_type == "final":
                    # Clean up internal fields if any
                    data.pop("action", None) 
                    final_data = data
                    
                    response_payload = {
                        "code": ErrorCode.SUCCESS.value,
                        "message": "ok",
                        "data": {**final_data, "session_id": session_id}
                    }
                    yield f"data: {json.dumps(response_payload, ensure_ascii=False)}\n\n"

            # 4. Save History
            new_history_entry_user = {"role": "user", "content": request.message, "user_id": request.user_id}
            assistant_content = final_data.get("reply") or full_reply

            if assistant_content:
                history.append(new_history_entry_user)
                history.append({"role": "assistant", "content": assistant_content})
                
                if len(history) > 60:
                    history = history[-60:]
                
                await redis_client.setex(redis_key, 86400, json.dumps(history))
            
            # Manual Logging of final response content
            process_time = (time.time() - start_time) * 1000
            formatted_process_time = "{0:.2f}".format(process_time)
            
            log_payload = {
                "code": ErrorCode.SUCCESS.value,
                "message": "ok",
                "data": {**final_data, "session_id": session_id}
            }
            logger.info(
                f"[Response] POST /api/v1/npc/chat/test/ - 响应数据: {json.dumps(log_payload, ensure_ascii=False)} - 耗时: {formatted_process_time} ms"
            )

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Stream failed in router")
            err_payload = {
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": f"Stream failed: {str(e)}",
                "data": None
            }
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


from src.llm.utils import call_api_stream_with_retry

@router.post("/npc/chat/stream_with_prompt/")
async def chat_with_prompt_stream(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Streaming endpoint that accepts a user-provided prompt and message.
    - Manually parses JSON body to avoid Pydantic dependency injection issues with StreamingResponse.
    - Streams events from the internal streaming service.
    - Does NOT log, persist history, or write to redis.
    """
    from fastapi import Request

    try:
        body = await request.json()
        message = body.get("message")
        prompt = body.get("prompt")
        
        if not message or not prompt:
            raise ValueError("Missing 'message' or 'prompt' in request body")
            
    except Exception as e:
        error_msg = str(e)
        async def err_gen():
            payload = {"code": ErrorCode.INTERNAL_ERROR.value, "message": f"invalid request: {error_msg}", "data": None}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def event_generator():
        try:
            # Construct messages for the LLM
            messages = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': message}
            ]

            logger.info(f"Starting prompt stream for session prompt_stream_{uuid.uuid4()}")
            
            # Using iterate_in_threadpool to consume sync generator safely
            generator = call_api_stream_with_retry(messages)
            
            async for chunk in iterate_in_threadpool(generator):
                if chunk:
                     yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Stream failed in prompt_stream")
            err_payload = {"code": ErrorCode.INTERNAL_ERROR.value, "message": f"Stream failed: {str(e)}", "data": None}
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Helper for Zodiac Calculation ---
def get_current_time_info() -> dict:
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(tz)
    hour = now.hour
    
    logger.info(f"[TimeDebug] Current server time: {now}, Hour: {hour}") # Add Logging for Debug

    # 1. Calculate Standard Zodiac (Time-based)
    # 子:23-1, 丑:1-3, ...
    # Format: (start_hour, end_hour, shichen, zodiac)
    zodiac_map = [
        (23, 1, "子时", "鼠"), (1, 3, "丑时", "牛"), (3, 5, "寅时", "虎"), (5, 7, "卯时", "兔"),
        (7, 9, "辰时", "龙"), (9, 11, "巳时", "蛇"), (11, 13, "午时", "马"), (13, 15, "未时", "羊"),
        (15, 17, "申时", "猴"), (17, 19, "酉时", "鸡"), (19, 21, "戌时", "狗"), (21, 23, "亥时", "猪")
    ]
    
    current_zodiac = "未知"
    current_shichen = "未知"
    
    for s, e, sh, z in zodiac_map:
        if s == 23: # Special case for 23:00 - 01:00
            if hour >= 23 or hour < 1:
                current_zodiac = z
                current_shichen = sh
                break
        else:
            if s <= hour < e:
                current_zodiac = z
                current_shichen = sh
                break
    
    # Simple Solar Term Approximation (sufficient for game context)
    # 2026 Solar Terms (Approximation)
    # 1.15 is roughly Xiao Han (Slight Cold) ~ Jan 5, Da Han (Great Cold) ~ Jan 20
    month = now.month
    day = now.day
    
    terms_map = {
        1: (5, "小寒", 20, "大寒"),
        2: (4, "立春", 18, "雨水"),
        3: (5, "惊蛰", 20, "春分"),
        4: (5, "清明", 20, "谷雨"),
        5: (5, "立夏", 21, "小满"),
        6: (5, "芒种", 21, "夏至"),
        7: (7, "小暑", 23, "大暑"),
        8: (7, "立秋", 23, "处暑"),
        9: (7, "白露", 23, "秋分"),
        10: (8, "寒露", 23, "霜降"),
        11: (7, "立冬", 22, "小雪"),
        12: (7, "大雪", 22, "冬至")
    }
    
    current_solar_term = "未知"
    if month in terms_map:
        d1, t1, d2, t2 = terms_map[month]
        if day < d1:
            # Previous month's second term needs proper lookup, simplifying to current month start
            current_solar_term = "节气交替" 
        elif day < d2:
            current_solar_term = t1
        else:
            current_solar_term = t2

    return {
        "time_str": now.strftime("%H:%M"),
        "date_str": now.strftime("%Y-%m-%d"),
        "zodiac": current_zodiac,
        "shichen": current_shichen,
        "solar_term": current_solar_term
    }

def fetch_rag_context(query: str) -> str:
    """Fetch knowledge from local RAG service."""
    if not query or len(query) > 100 or len(query.strip()) < 2:
        return "无相关资料"
    
    rag_url = "http://localhost:7002/smart_query/"
    try:
        # Short timeout to ensure main chat doesn't lag
        response = requests.get(rag_url, params={"query": query}, timeout=3.0)
        if response.status_code == 200:
            try:
                # Try parsing as JSON first
                data = response.json()
                
                # 1. User Defined Format (status="success", results=[...])
                if isinstance(data, dict) and data.get("status") == "success":
                    results = data.get("results", [])
                    if not results:
                        return "未检索到相关信息"
                        
                    context_pieces = []
                    for i, item in enumerate(results[:3]): # Limit to top 3 chunks
                        content = item.get("content", "").strip()
                        metadata = item.get("metadata", {})
                        title = metadata.get("title", "")
                        
                        if content:
                            source_info = f" (来源: {title})" if title else ""
                            context_pieces.append(f"{i+1}. {content}{source_info}")
                            
                    return "\n\n".join(context_pieces) if context_pieces else "检索结果内容为空"

                # 2. Fallback Logic for other structures
                if isinstance(data, dict):
                    # Priority 1: Direct answer fields
                    answer = data.get("answer") or data.get("result") or data.get("generated_text")
                    if answer and isinstance(answer, str):
                        return answer
                        
                    # Priority 2: List of retrieval chunks (docs/chunks/data)
                    docs = data.get("docs") or data.get("documents") or data.get("chunks") or data.get("data")
                    if isinstance(docs, list):
                        context_pieces = []
                        for i, doc in enumerate(docs[:3]):
                             if isinstance(doc, dict):
                                 txt = doc.get("content") or doc.get("text") or str(doc)
                                 context_pieces.append(f"{i+1}. {txt}")
                             else:
                                 context_pieces.append(f"{i+1}. {str(doc)}")
                        return "\n".join(context_pieces) if context_pieces else str(data)

                return str(data)[:1000]
            except Exception as e:
                logger.warning(f"RAG parsing error: {e}")
                # Fallback to text
                return response.text[:500] 
    except Exception as e:
        logger.warning(f"RAG Service unavailable: {e}")
    return "暂时无法连接知识库"