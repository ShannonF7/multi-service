from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from sqlalchemy.orm import Session
import uuid
import logging
import json
import time
import io
import time
from PIL import Image

from src.database.session import get_db
from src.database.redis import get_redis
from src.database.models import ScriptTemplate, GeneratedScript
from src.database.models_existing import Attraction, AttractionImage
from src.scripts.service import generate_full_dynamic_script, process_chat, process_assistant_chat, process_chat_stream, process_assistant_chat_stream
from src.scripts.service import _NPC_CONFIRM_PENDING_SENTINEL, _should_enter_confirm_pending
from src.scripts.schemas import BaseResponse, GenerateScriptRequest, ErrorCode, ChatRequest
from src.cv.feature_extractor import get_feature_extractor

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

import time

@router.post("/task/verify-image/")
async def verify_image(
    file: UploadFile = File(...),
    target_attraction_name: str = Form(None),
    db: Session = Depends(get_db)
):
    try:
        # 1. Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 2. Extract features (run in threadpool to avoid blocking event loop)
        extractor = get_feature_extractor()
        feature_vector = await run_in_threadpool(extractor.extract, image)
        
        if not feature_vector:
             logger.warning("Failed to extract features from uploaded image.")
             raise HTTPException(status_code=400, detail="Failed to extract features from image")

        # 3. Search in Database
        # Using L2 distance (Euclidean distance) for similarity
        # Note: SimCLR features are normalized, so L2 distance is related to Cosine Similarity
        # Lower distance = Higher similarity
        nearest_image = db.query(AttractionImage).order_by(
            AttractionImage.embedding.l2_distance(feature_vector)
        ).limit(1).first()

        response_data = None
        if not nearest_image:
            response_data = {
                "match": False,
                "identified_attraction": None,
                "target_attraction": target_attraction_name
            }
            response_obj = BaseResponse(
                code=200, 
                message="No matching attraction found", 
                data=response_data
            )
        else:
            # 4. Get Attraction Details
            attraction = db.query(Attraction).filter(Attraction.id == nearest_image.attraction_id).first()
            identified_name = attraction.name if attraction else "Unknown"
            
            # 5. Verify against target if provided
            is_verified = True
            if target_attraction_name:
                # Simple containment check for robustness
                is_verified = (target_attraction_name in identified_name) or (identified_name in target_attraction_name)

            response_data = {
                "match": is_verified,
                "identified_attraction": identified_name,
                "target_attraction": target_attraction_name,
            }
            response_obj = BaseResponse(
                code=200,
                message="Success",
                data=response_data
            )

        return response_obj

    except Exception as e:
        logger.error(f"Error in verify_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-script/", response_model=BaseResponse)
async def generate_script(
    request: GenerateScriptRequest,
    db: Session = Depends(get_db)
):
    try:
        # Fetch template from DB
        template_record = db.query(ScriptTemplate).filter(
            ScriptTemplate.id == request.template_id,
            ScriptTemplate.is_active == True
        ).first()

        if not template_record:
             raise HTTPException(
                status_code=404,
                detail=f"No active script template found for id: {request.template_id}"
            )

        dynamic_script = generate_full_dynamic_script(
            team_id=request.team_id,
            script_template_data=template_record.template,
            style=request.style,
            db=db
        )

        # Save to GeneratedScript
        new_script = GeneratedScript(
            team_id=uuid.UUID(request.team_id),
            template_id=template_record.id,
            script=dynamic_script,
            status='generated'
        )
        db.add(new_script)
        db.commit()
        db.refresh(new_script)

        return BaseResponse(
            code=ErrorCode.SUCCESS.value,
            message="ok",
            data={
                "generated_script_id": new_script.id
            }
        )

    except Exception as e:
        logger.exception("Script generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Script generation failed: {str(e)}"
        )

@router.post("/npc/chat/", response_model=BaseResponse)
async def chat_with_npc(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    try:
        # Redis Session Management
        redis_client = await get_redis()
        session_id = request.session_id or str(uuid.uuid4())
        redis_key = f"chat_session:{session_id}"
        
        # Load history from Redis
        history = []
        if request.session_id:
            cached_history = await redis_client.get(redis_key)
            if cached_history:
                try:
                    history = json.loads(cached_history)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode history for session {session_id}")
                    history = []
        
        # Fallback/Merge: If Redis empty, use client history (optional, for migration)
        if not history and request.history:
            history = request.history

        # Run synchronous process_chat in a thread pool to avoid blocking the event loop
        result = await run_in_threadpool(
            process_chat,
            team_id=request.team_id,
            user_id=request.user_id,
            task_id=request.task_id,
            message=request.message,
            history=history,
            generated_script_id=request.generated_script_id,
            task_status=request.task_status,
            sub_task_id=request.sub_task_id,
            db=db,
            image_result=request.image_result
        )
        
        # Update History
        new_history_entry_user = {"role": "user", "content": request.message, "user_id": request.user_id}

        assistant_content = result["reply"]
        if _should_enter_confirm_pending(
            completion_mechanism="NPC_DIALOGUE_COMPLETE",
            user_message=request.message,
            task_completed=bool(result.get("task_completed")),
        ):
            assistant_content = (assistant_content or "").rstrip() + f"\n{_NPC_CONFIRM_PENDING_SENTINEL}"

        new_history_entry_npc = {"role": "assistant", "content": assistant_content}
        
        history.append(new_history_entry_user)
        history.append(new_history_entry_npc)
        
        # Keep only last 30 turns (60 messages) - Balanced for cost and context
        if len(history) > 60:
            history = history[-60:]
            
        # Save to Redis (24 hours expiration)
        await redis_client.setex(redis_key, 86400, json.dumps(history))
        
        # Construct clean response
        # Note: process_chat result is filtered here to ensure strict API contract
        # 对外回包不展示哨兵标记
        public_reply = (result.get("reply") or "").replace(f"\n{_NPC_CONFIRM_PENDING_SENTINEL}", "")

        final_result = {
            "reply": public_reply,
            "task_completed": result.get("task_completed"),
            "action": result.get("action"),
            "action_data": result.get("action_data"),
            "session_id": session_id
        }
        
        return BaseResponse(
            code=ErrorCode.SUCCESS.value,
            message="ok",
            data=final_result
        )
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )



# @router.post("/npc/chat/stream/")
# async def chat_with_npc_stream(
#     request: ChatRequest,
#     db: Session = Depends(get_db)
# ):
#     try:
#         req_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
#         req_dict.pop("history", None)
#         logger.info(
#             f"[Request] POST /api/v1/npc/chat/stream - 请求数据: {json.dumps(req_dict, ensure_ascii=False)}"
#         )
#     except Exception:
#         logger.info("[Request] POST /api/v1/npc/chat/stream - 请求数据: <Request Serialize Failed>")

#     redis_client = await get_redis()
#     session_id = request.session_id or str(uuid.uuid4())
#     redis_key = f"chat_session:{session_id}"

#     # Load history from Redis
#     history = []
#     if request.session_id:
#         cached_history = await redis_client.get(redis_key)
#         if cached_history:
#             try:
#                 history = json.loads(cached_history)
#             except json.JSONDecodeError:
#                 logger.warning(f"Failed to decode history for session {session_id}")
#                 history = []

#     if not history and request.history:
#         history = request.history

#     async def event_gen():
#         start_time = time.time()
#         local_history = list(history) if history else []
#         local_history.append({"role": "user", "content": request.message, "user_id": request.user_id})

#         full_text_holder = {"text": ""}
#         final_struct = {"reply": "", "task_completed": False, "action": "NONE", "action_data": {}}

#         try:
#             def run_and_collect_events():
#                 for evt_type, payload in process_chat_stream(
#                     team_id=request.team_id,
#                     user_id=request.user_id,
#                     task_id=request.task_id,
#                     message=request.message,
#                     history=history,
#                     generated_script_id=request.generated_script_id,
#                     task_status=request.task_status,
#                     sub_task_id=request.sub_task_id,
#                     db=db,
#                     image_result=request.image_result,
#                 ):
#                     yield (evt_type, payload)

#             async for evt_type, payload in iterate_in_threadpool(run_and_collect_events()):
#                 if evt_type == "delta":
#                     full_text_holder["text"] += payload
#                     data = json.dumps({"text": payload}, ensure_ascii=False)
#                     yield f"event: delta\ndata: {data}\n\n"
#                 elif evt_type == "final":
#                     final_struct.update({
#                         "reply": payload.get("reply", ""),
#                         "task_completed": payload.get("task_completed", False),
#                         "action": payload.get("action", "NONE"),
#                         "action_data": payload.get("action_data", {}),
#                     })

#             assistant_content = final_struct["reply"]
#             if _should_enter_confirm_pending(
#                 completion_mechanism="NPC_DIALOGUE_COMPLETE",
#                 user_message=request.message,
#                 task_completed=bool(final_struct.get("task_completed")),
#             ):
#                 assistant_content = (assistant_content or "").rstrip() + f"\n{_NPC_CONFIRM_PENDING_SENTINEL}"

#             local_history.append({"role": "assistant", "content": assistant_content})
#             if len(local_history) > 60:
#                 local_history = local_history[-60:]
#             await redis_client.setex(redis_key, 86400, json.dumps(local_history, ensure_ascii=False))

#             process_time = (time.time() - start_time) * 1000
#             formatted_process_time = "{0:.2f}".format(process_time)

#             # 对外回包不展示哨兵标记
#             public_struct = dict(final_struct)
#             public_struct["reply"] = (public_struct.get("reply") or "").replace(f"\n{_NPC_CONFIRM_PENDING_SENTINEL}", "")

#             final_data = json.dumps({
#                 **public_struct,
#                 "session_id": session_id,
#             }, ensure_ascii=False)

#             logger.info(
#                 f"[Response] POST /api/v1/npc/chat/stream - 响应数据: {final_data} - 耗时: {formatted_process_time} ms"
#             )
#             yield f"event: final\ndata: {final_data}\n\n"

#         except Exception as e:
#             logger.exception("Chat stream failed")
#             err = json.dumps({
#                 "code": ErrorCode.INTERNAL_ERROR.value,
#                 "message": f"Chat stream failed: {str(e)}",
#                 "session_id": session_id,
#             }, ensure_ascii=False)
#             yield f"event: error\ndata: {err}\n\n"

#     return StreamingResponse(
#         event_gen(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no",
#         },
#     )

@router.post("/assistant/chat/", response_model=BaseResponse)
async def chat_with_assistant(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    try:
        # Redis Session Management for Assistant
        redis_client = await get_redis()
        # Use a distinct prefix for assistant chat history
        session_id = request.session_id or str(uuid.uuid4())
        redis_key = f"assistant_session:{session_id}"
        
        # Load history from Redis
        history = []
        if request.session_id:
            cached_history = await redis_client.get(redis_key)
            if cached_history:
                try:
                    history = json.loads(cached_history)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode assistant history for session {session_id}")
                    history = []
        
        # Fallback/Merge
        if not history and request.history:
            history = request.history

        # Run synchronous process_assistant_chat in a thread pool
        result = await run_in_threadpool(
            process_assistant_chat,
            team_id=request.team_id,
            user_id=request.user_id,
            task_id=request.task_id,
            message=request.message,
            history=history,
            db=db,
            generated_script_id=request.generated_script_id,
            task_status=request.task_status,
            sub_task_id=request.sub_task_id
        )

        # Update History
        # Append User Message
        history.append({"role": "user", "content": request.message, "user_id": request.user_id})
        # Append Assistant Reply
        history.append({"role": "assistant", "content": result["reply"]})
        
        # Save to Redis (24h expiry)
        await redis_client.setex(redis_key, 86400, json.dumps(history))

        return BaseResponse(
            code=ErrorCode.SUCCESS.value,
            message="ok",
            data={
                "reply": result["reply"],
                "task_completed": result["task_completed"],
                "action": result["action"],
                "session_id": session_id
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Assistant chat failed")
        raise HTTPException(
            status_code=500,
            detail=f"Assistant chat failed: {str(e)}"
        )



@router.post("/assistant/chat/stream/")
async def chat_with_assistant_stream(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    try:
        req_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        req_dict.pop("history", None)
        logger.info(
            f"[Request] POST /api/v1/assistant/chat/stream - 请求数据: {json.dumps(req_dict, ensure_ascii=False)}"
        )
    except Exception:
        logger.info("[Request] POST /api/v1/assistant/chat/stream - 请求数据: <Request Serialize Failed>")

    redis_client = await get_redis()
    session_id = request.session_id or str(uuid.uuid4())
    redis_key = f"assistant_session:{session_id}"

    history = []
    if request.session_id:
        cached_history = await redis_client.get(redis_key)
        if cached_history:
            try:
                history = json.loads(cached_history)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode assistant history for session {session_id}")
                history = []

    if not history and request.history:
        history = request.history

    async def event_gen():
        start_time = time.time()
        local_history = list(history) if history else []
        local_history.append({"role": "user", "content": request.message, "user_id": request.user_id})

        final_reply_text = {"text": ""}

        try:
            def run_and_collect_events():
                for evt_type, payload in process_assistant_chat_stream(
                    team_id=request.team_id,
                    user_id=request.user_id,
                    task_id=request.task_id,
                    message=request.message,
                    history=history,
                    db=db,
                    generated_script_id=request.generated_script_id,
                    task_status=request.task_status,
                    sub_task_id=request.sub_task_id,
                ):
                    yield (evt_type, payload)

            async for evt_type, payload in iterate_in_threadpool(run_and_collect_events()):
                if evt_type == "delta":
                    final_reply_text["text"] += payload
                    data = json.dumps({"text": payload}, ensure_ascii=False)
                    yield f"event: delta\ndata: {data}\n\n"
                elif evt_type == "final":
                    # payload: reply/full_text
                    final_reply_text["text"] = payload.get("reply", final_reply_text["text"])

            local_history.append({"role": "assistant", "content": final_reply_text["text"]})
            if len(local_history) > 60:
                local_history = local_history[-60:]
            await redis_client.setex(redis_key, 86400, json.dumps(local_history, ensure_ascii=False))

            final_data = json.dumps({
                "reply": final_reply_text["text"],
                "task_completed": False,
                "action": "NONE",
                "session_id": session_id,
            }, ensure_ascii=False)

            process_time = (time.time() - start_time) * 1000
            formatted_process_time = "{0:.2f}".format(process_time)
            logger.info(
                f"[Response] POST /api/v1/assistant/chat/stream - 响应数据: {final_data} - 耗时: {formatted_process_time} ms"
            )
            yield f"event: final\ndata: {final_data}\n\n"

        except Exception as e:
            logger.exception("Assistant chat stream failed")
            err = json.dumps({
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": f"Assistant chat stream failed: {str(e)}",
                "session_id": session_id,
            }, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )