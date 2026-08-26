from typing import Dict, Any, Optional, List
import json
import logging
from sqlalchemy.orm import Session
from src.llm import dialogue_eval
from src.llm.dialogue_eval import TaskState, run_task_pipeline
from src.scripts.schemas import ChatRequest
from src.database.models import GeneratedScript
from src.scripts.service import process_chat_v2, process_chat_v2_test
from src.llm.utils import call_api_stream_with_retry, call_api_with_retry, get_zodiac_from_hour
from src.llm.prompts.prompts import (
    npc_chat_template_v2, 
    zhangbi_game_prompt_v2,
    npc_core_prompt,
    judge_prompt,
    turtle_soup_prompt
)
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def handle_dialogue_request(
    request: ChatRequest,
    db: Session
) -> Dict[str, Any]:
    """
    Unified Dialogue Entry Point with DB support.
    """
    
    # 1. Fetch Script and Task Config from DB to determine Task Type & Context
    # We need to minimally load the script to find the current task configuration
    # for evaluation purposes.
    
    task_config = None
    task_type = "UNKNOWN"
    
    # Try to load script record
    script_record = db.query(GeneratedScript).filter(
        GeneratedScript.team_id == request.team_id
    ).order_by(GeneratedScript.created_at.desc()).first()
      
    if script_record and script_record.script:
        script_content = script_record.script
        tasks = script_content.get("tasks", [])
        current_task = next((t for t in tasks if t["task_id"] == request.task_id), None)
        
        if current_task:
            task_type = current_task.get("task_type", "NPC_INTERACTION")
            task_config = current_task
            # Ensure triggers/criteria exist in config if we want to eval
    
    # 2. Evaluation Logic (Only for NPC_INTERACTION)
    if task_type == "NPC_INTERACTION" and task_config:
        # Determine previous state
        status_str = request.task_status or "in_progress"
        try:
            prev_state = TaskState(status_str.upper())
        except ValueError:
            prev_state = TaskState.IN_PROGRESS

        # Run Evaluation Pipeline
        try:
            pipeline_result = run_task_pipeline(
                task_id=request.task_id,
                user_text=request.message,
                prev_state=prev_state,
                task_config=task_config
            )
            
            rule_decision = pipeline_result.rule
            eval_result = pipeline_result.eval_result
            
            # Check if evaluation triggered a specific response
            if eval_result.is_task_related:
                # 评估结果始终使用规则引擎返回的 message 作为 reply（包含提示/拒绝/完成）
                if rule_decision.message:
                    return {
                        "reply": rule_decision.message,
                        "task_completed": bool(rule_decision.task_completed),
                    }

                # 无明确 message 时，仅在完成时给默认祝贺语
                if rule_decision.task_completed:
                    return {
                        "reply": "恭喜，你完成了任务！",
                        "task_completed": True,
                    }
        except Exception as e:
            logger.error(f"Error during dialogue evaluation: {e}")
            # Continue to fallback if eval fails

    # 3. Fallback / General Dialogue Generation
    # Reuse the robust existing logic in src/scripts/service.py
    raw = process_chat_v2(
        team_id=request.team_id,
        user_id=request.user_id,
        task_id=request.task_id,
        message=request.message,
        history=request.history or [],
        db=db,
        generated_script_id=request.generated_script_id,
        task_status=request.task_status,
        sub_task_id=request.sub_task_id,
        image_result=request.image_result
    )

    return {
        "reply": raw.get("reply", ""),
        "task_completed": bool(raw.get("task_completed", False)),
    }

def handle_dialogue_request_test(
    request: ChatRequest,
    db: Session
) -> Dict[str, Any]:
    """
    Unified Dialogue Entry Point with DB support.
    """
    
    # 1. Fetch Script and Task Config from DB to determine Task Type & Context
    # We need to minimally load the script to find the current task configuration
    # for evaluation purposes.
    
    task_config = None
    task_type = "UNKNOWN"
    
    # Try to load script record
    script_record = db.query(GeneratedScript).filter(
        GeneratedScript.team_id == request.team_id
    ).order_by(GeneratedScript.created_at.desc()).first()
      
    if script_record and script_record.script:
        script_content = script_record.script
        tasks = script_content.get("tasks", [])
        current_task = next((t for t in tasks if t["task_id"] == request.task_id), None)
        
        if current_task:
            task_type = current_task.get("task_type", "NPC_INTERACTION")
            task_config = current_task
            # Ensure triggers/criteria exist in config if we want to eval
    
    # 2. Evaluation Logic (Only for NPC_INTERACTION)
    if task_type == "NPC_INTERACTION" and task_config:
        # Determine previous state
        status_str = request.task_status or "in_progress"
        try:
            prev_state = TaskState(status_str.upper())
        except ValueError:
            prev_state = TaskState.IN_PROGRESS

        # Run Evaluation Pipeline
        try:
            pipeline_result = run_task_pipeline(
                task_id=request.task_id,
                user_text=request.message,
                prev_state=prev_state,
                task_config=task_config
            )
            
            rule_decision = pipeline_result.rule
            eval_result = pipeline_result.eval_result
            
            # Check if evaluation triggered a specific response
            if eval_result.is_task_related:
                # 评估结果始终使用规则引擎返回的 message 作为 reply（包含提示/拒绝/完成）
                if rule_decision.message:
                    return {
                        "reply": rule_decision.message,
                        "task_completed": bool(rule_decision.task_completed),
                    }

                # 无明确 message 时，仅在完成时给默认祝贺语
                if rule_decision.task_completed:
                    return {
                        "reply": "恭喜，你完成了任务！",
                        "task_completed": True,
                    }
        except Exception as e:
            logger.error(f"Error during dialogue evaluation: {e}")
            # Continue to fallback if eval fails

    # 3. Fallback / General Dialogue Generation
    # Reuse the robust existing logic in src/scripts/service.py
    raw = process_chat_v2_test(
        team_id=request.team_id,
        user_id=request.user_id,
        task_id=request.task_id,
        message=request.message,
        history=request.history or [],
        db=db,
        generated_script_id=request.generated_script_id,
        task_status=request.task_status,
        sub_task_id=request.sub_task_id,
        image_result=request.image_result
    )

    return {
        "reply": raw.get("reply", ""),
        "task_completed": bool(raw.get("task_completed", False)),
    }

def _simple_parse_json(text: str) -> Dict[str, Any]:
    try:
        clean = text.strip()
        # Handle potential <think> tags from thinking models
        import re
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL).strip()
        
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]
        
        # If still not clean JSON, try to extract from first { to last }
        start = clean.find('{')
        end = clean.rfind('}')
        if start != -1 and end != -1:
            clean = clean[start:end+1]
            
        return json.loads(clean)
    except:
        logger.warning(f"JSON Parse Failed. Raw text: {text}")
        return {"reply": text, "task_completed": False} 

async def handle_dialogue_stream(
    team_id: str,
    user_id: str,
    task_id: str,
    message: str,
    history: List[Dict],
    db: Session,
    generated_script_id: str = None,
    task_status: str = "in_progress",
    sub_task_id: str = None,
    image_result: Dict = None,
):
    """
    Streaming version of handle_dialogue_request with Evaluation support.
    """
    # 1. Fetch Script and Task Config
    task_config = None
    task_type = "UNKNOWN"
    
    query = db.query(GeneratedScript).filter(GeneratedScript.team_id == team_id)
    if generated_script_id:
        query = query.filter(GeneratedScript.id == generated_script_id)
    script_record = query.order_by(GeneratedScript.created_at.desc()).first()
    
    if not script_record:
        # If no script, we can't do much. 
        yield ("delta", "错误：未找到剧本配置。")
        yield ("final", {"reply": "错误：未找到剧本配置。", "task_completed": False})
        return

    script_content = script_record.script or {}
    tasks = script_content.get("tasks", [])
    current_task = next((t for t in tasks if t["task_id"] == task_id), None)
    
    # If the requested task_id is not present in the loaded script, return a clear error
    if not current_task:
        yield ("delta", f"错误：在当前剧本中未找到任务 {task_id}。请确认 generated_script_id 或 task_id 是否正确。")
        yield ("final", {"reply": f"错误：在当前剧本中未找到任务 {task_id}。", "task_completed": False})
        return

    if current_task:
        task_type = current_task.get("task_type", "NPC_INTERACTION")
        task_config = current_task
        # Fallback for triggers if missing
        if task_type == "NPC_INTERACTION":
             if not task_config.get("triggers") or not task_config.get("completion_criteria"):
                try:
                    template_conf = dialogue_eval.load_task_config(task_id)
                    if not task_config.get("triggers"):
                        task_config["triggers"] = template_conf.get("triggers", [])
                    if not task_config.get("completion_criteria"):
                        task_config["completion_criteria"] = template_conf.get("completion_criteria", {})
                except Exception:
                    pass

    # 2. Evaluation Logic (Only for NPC_INTERACTION)
    eval_triggered_reply = None
    eval_task_completed = False
    
    if task_type == "NPC_INTERACTION" and task_config:
        status_str = task_status or "in_progress"
        try:
            prev_state = TaskState(status_str.upper())
        except ValueError:
            prev_state = TaskState.IN_PROGRESS

        try:
            # Sync call - might block briefly, but usually fast enough
            pipeline_result = run_task_pipeline(
                task_id=task_id,
                user_text=message,
                prev_state=prev_state,
                task_config=task_config
            )
            
            rule_decision = pipeline_result.rule
            eval_result = pipeline_result.eval_result
            
            if eval_result.is_task_related:
                if rule_decision.message:
                    eval_triggered_reply = rule_decision.message
                    eval_task_completed = bool(rule_decision.task_completed)
                elif rule_decision.task_completed:
                    eval_triggered_reply = "恭喜，你完成了任务！"
                    eval_task_completed = True
                    
        except Exception as e:
            logger.error(f"Error during dialogue evaluation stream: {e}")

    # If Eval produced a result, yield it as a stream
    if eval_triggered_reply:
        # Mock streaming for consistent UX
        chunk_size = 5
        for i in range(0, len(eval_triggered_reply), chunk_size):
            yield ("delta", eval_triggered_reply[i:i+chunk_size])
            import asyncio
            # In async generator, we can't easily sleep if this isn't async def. 
            # But the router will consume this iterator.
            # Assuming fast return is fine.
        
        yield ("final", {
            "reply": eval_triggered_reply, 
            "task_completed": eval_task_completed
        })
        return

    # 3. Fallback: LLM Generation (Streaming)
    # Build prompt context - Logic mirrored from process_chat_v2
    script_meta = script_content.get("script_metadata", {})
    script_name = script_meta.get("title", "")
    style = script_meta.get("generated_script_style", "")
    era_background = script_meta.get("era_background", "")
    mechanism = script_content.get("mechanism", "")

    npc_name = current_task.get("npc") or "神秘向导"
    if "{{" in str(npc_name): npc_name = "神秘向导"
    
    npc_role = current_task.get("npc_role") or "指引者"
    if "{{" in str(npc_role): npc_role = "指引者"
    
    stage_name = current_task.get("stage_name") or ""
    objective = current_task.get("objective_template") or ""
    location = current_task.get("location") or ""
    selected_item = current_task.get("selected_item") or "无"
    initial_dialogue = current_task.get("npc_dialogue") or "无"
    completion_mechanism = current_task.get("completion_mechanism") or "NPC_DIALOGUE_COMPLETE"

    # Subtasks
    sub_tasks_info = "无子任务"
    sub_tasks = current_task.get("sub_tasks") or []
    active_sub_task = None
    
    # [FEATURE] Focused Subtask Context
    # If sub_task_id is provided, we should focus the NPC context on that specific subtask.
    
    if sub_tasks:
        sub_tasks_info = "当前任务包含以下子任务（请按顺序引导用户完成）：\n"
        for i, st in enumerate(sub_tasks):
            s_id = st.get("sub_task_id")
            s_loc = st.get("location") or "周边区域"
            is_current = (s_id == sub_task_id)
            if is_current:
                 active_sub_task = st
                 # Override context with active subtask info
                 if st.get("location"):
                     location = f"{location} -> {st.get('location')}"
                 
                 # Focus the objective on the subtask
                 sub_mech = st.get("completion_mechanism")
                 if isinstance(sub_mech, list) and len(sub_mech) > 0: sub_mech = sub_mech[0]
                 
                 # Append specific instructions to objective
                 objective += f"\n\n[当前专注子任务]: {st.get('game', s_id)}"
                 if st.get("task_data"):
                     td = st.get("task_data")
                     if "target_photo_description" in td:
                         objective += f"\n[拍摄指引]: {td['target_photo_description']}"
                     if "description" in td:
                         objective += f"\n[任务指引]: {td['description']}"
            
            status_mark = "[进行中]" if is_current else "[待完成]"
            
            task_data = st.get("task_data") or {}
            completion_mech = st.get("completion_mechanism", "UNKNOWN")
            
            if completion_mech == "AI_IMAGE_JUDGE":
                target_name = task_data.get("target_photo_name", "指定目标")
                raw_desc = task_data.get("target_photo_description", "请拍摄该目标")
                description = f"【拍摄目标】：{target_name} (地点: {s_loc})。 \n【背景引导】：{raw_desc}"
                description += " (⚠️系统要求：需等待 [系统通知: 图片验证通过] 才能判定完成。NPC请明确告知用户需要拍摄什么。)"
            elif completion_mech == "AI_ANSWER_CORRECT":
                description = task_data.get("ai_answer_prompt", "") or f"问题: {task_data.get('ai_question', '')}"
                description += f" (地点: {s_loc})"
            else:
                description = task_data.get("description", "请完成此步骤") + f" (地点: {s_loc})"

            reward = st.get("virtual_reward")
            reward_str = f" [奖励: {reward.get('item_semantic', '')}]" if reward else ""
            
            sub_tasks_info += f"{i+1}. {status_mark} 子任务ID: {s_id}, 名称: {st.get('game')}, 机制: {completion_mech}, 地点: {s_loc}, 描述: {description}{reward_str}\n"

    # History
    history_text = ""
    if history:
        for h in history:
            role = h.get("role")
            content = h.get("content", "")
            if role == "user":
                history_text += f"[用户]: {content}\n"
            else:
                history_text += f"[NPC]: {content}\n"

    # Input
    safe_message = message.replace("[系统通知", "")
    user_input_with_id = f"[用户ID: {user_id}] 说: {safe_message}"
    if sub_task_id:
        user_input_with_id += f" (当前正在进行子任务: {sub_task_id})"
    
    system_notification = "无"
    if image_result:
         status = "通过" if image_result.get("success") else "失败"
         reason = image_result.get("message", "无")
         system_notification = f"[系统通知: 图片验证{status}，原因: {reason}]"

    prompt = npc_chat_template_v2.format(
        script_name=script_name,
        npc_name=npc_name,
        npc_role=npc_role,
        style=style,
        era_background=era_background,
        mechanism=mechanism,
        stage_name=stage_name,
        objective=objective,
        task_type=task_type,
        completion_mechanism=completion_mechanism,
        selected_item=selected_item,
        initial_dialogue=initial_dialogue,
        task_status=task_status,
        location=location,
        sub_tasks_info=sub_tasks_info,
        history=history_text,
        system_notification=system_notification,
        user_input=user_input_with_id,
    )

    messages = [{"role": "system", "content": prompt}]
    
    full_text = ""
    # Streaming call
    for part in call_api_stream_with_retry(messages):
        full_text += part
        yield ("delta", part)
        
    final_payload = _simple_parse_json(full_text)
    yield ("final", final_payload)


async def handle_dialogue_stream_test(
    team_id: str,
    user_id: str,
    task_id: str,
    message: str,
    history: List[Dict],
    db: Session,
    generated_script_id: str = None,
    task_status: str = "in_progress",
    sub_task_id: str = None,
    image_result: Dict = None,
    rag_context: str = "无",
    time_info: Dict = None
):
    """
    Streaming version of handle_dialogue_request with Evaluation support.
    """
    # 1. Fetch Script and Task Config
    task_config = None
    task_type = "UNKNOWN"
    
    query = db.query(GeneratedScript).filter(GeneratedScript.team_id == team_id)
    if generated_script_id:
        query = query.filter(GeneratedScript.id == generated_script_id)
    script_record = query.order_by(GeneratedScript.created_at.desc()).first()
    
    if not script_record:
        # If no script, we can't do much. 
        yield ("delta", "错误：未找到剧本配置。")
        yield ("final", {"reply": "错误：未找到剧本配置。", "task_completed": False})
        return

    script_content = script_record.script or {}
    tasks = script_content.get("tasks", [])
    current_task = next((t for t in tasks if t["task_id"] == task_id), None)

    # If the requested task_id is not present in the loaded script, return a clear error
    if not current_task:
        yield ("delta", f"错误：在当前剧本中未找到任务 {task_id}。请确认 generated_script_id 或 task_id 是否正确。")
        yield ("final", {"reply": f"错误：在当前剧本中未找到任务 {task_id}。", "task_completed": False})
        return

    if current_task:
        task_type = current_task.get("task_type", "NPC_INTERACTION")
        task_config = current_task
        # Fallback for triggers if missing
        if task_type == "NPC_INTERACTION":
             if not task_config.get("triggers") or not task_config.get("completion_criteria"):
                try:
                    template_conf = dialogue_eval.load_task_config(task_id)
                    if not task_config.get("triggers"):
                        task_config["triggers"] = template_conf.get("triggers", [])
                    if not task_config.get("completion_criteria"):
                        task_config["completion_criteria"] = template_conf.get("completion_criteria", {})
                except Exception:
                    pass

                except Exception:
                    pass

    # 2. Evaluation Logic (Only for NPC_INTERACTION)
    eval_triggered_reply = None
    eval_task_completed = False
    
    if task_type == "NPC_INTERACTION" and task_config:
        status_str = task_status or "in_progress"
        try:
            prev_state = TaskState(status_str.upper())
        except ValueError:
            prev_state = TaskState.IN_PROGRESS

        try:
            # Sync call - might block briefly, but usually fast enough
            pipeline_result = run_task_pipeline(
                task_id=task_id,
                user_text=message,
                prev_state=prev_state,
                task_config=task_config
            )
            
            rule_decision = pipeline_result.rule
            eval_result = pipeline_result.eval_result
            
            if eval_result.is_task_related:
                if rule_decision.message:
                    eval_triggered_reply = rule_decision.message
                    eval_task_completed = bool(rule_decision.task_completed)
                elif rule_decision.task_completed:
                    eval_triggered_reply = "恭喜，你完成了任务！"
                    eval_task_completed = True
                    
        except Exception as e:
            logger.error(f"Error during dialogue evaluation stream: {e}")

    # If Eval produced a result, yield it as a stream
    if eval_triggered_reply:
        # Mock streaming for consistent UX
        chunk_size = 5
        for i in range(0, len(eval_triggered_reply), chunk_size):
            yield ("delta", eval_triggered_reply[i:i+chunk_size])
            import asyncio
            # In async generator, we can't easily sleep if this isn't async def. 
            # But the router will consume this iterator.
            # Assuming fast return is fine.
        
        yield ("final", {
            "reply": eval_triggered_reply, 
            "task_completed": eval_task_completed
        })
        return

    # 3. Fallback: LLM Generation (Streaming)
    # Build prompt context - Logic mirrored from process_chat_v2
    script_meta = script_content.get("script_metadata", {})
    script_name = script_meta.get("title", "")
    style = script_meta.get("generated_script_style", "")
    era_background = script_meta.get("era_background", "")
    mechanism = script_content.get("mechanism", "")

    npc_name = current_task.get("npc") or "神秘向导"
    if "{{" in str(npc_name): npc_name = "神秘向导"
    
    npc_role = current_task.get("npc_role") or "指引者"
    if "{{" in str(npc_role): npc_role = "指引者"
    
    stage_name = current_task.get("stage_name") or ""
    objective = current_task.get("objective_template") or ""
    location = current_task.get("location") or ""
    selected_item = current_task.get("selected_item") or "无"
    initial_dialogue = current_task.get("npc_dialogue") or "无"
    completion_mechanism = current_task.get("completion_mechanism") or "NPC_DIALOGUE_COMPLETE"
    # [FIX] Normalize completion_mechanism if it is a list (e.g. ["AI_NPC_DIALOGUE_COMPLETE"])
    if isinstance(completion_mechanism, list):
        if len(completion_mechanism) > 0:
            completion_mechanism = completion_mechanism[0]
        else:
            completion_mechanism = "UNKNOWN"

    task_knowledge = current_task.get("task_knowledge") or "无"

    # Subtasks
    sub_tasks_info = "无子任务"
    sub_tasks = current_task.get("sub_tasks") or []
    active_sub_task = None
    
    # [FEATURE] Focused Subtask Context
    # If sub_task_id is provided, we should focus the NPC context on that specific subtask.
    
    if sub_tasks:
        sub_tasks_info = "当前任务包含以下子任务（请按顺序引导用户完成）：\n"
        for i, st in enumerate(sub_tasks):
            s_id = st.get("sub_task_id")
            s_loc = st.get("location") or "周边区域"
            is_current = (s_id == sub_task_id)
            if is_current:
                 active_sub_task = st
                 # Override context with active subtask info
                 if st.get("location"):
                     location = f"{location} -> {st.get('location')}"
                 
                 # Focus the objective on the subtask
                 sub_mech = st.get("completion_mechanism")
                 if isinstance(sub_mech, list) and len(sub_mech) > 0: sub_mech = sub_mech[0]
                 
                 # Append specific instructions to objective
                 objective += f"\n\n[当前专注子任务]: {st.get('game', s_id)}"
                 if st.get("task_data"):
                     td = st.get("task_data")
                     if "target_photo_description" in td:
                         objective += f"\n[拍摄指引]: {td['target_photo_description']}"
                     if "description" in td:
                         objective += f"\n[任务指引]: {td['description']}"
            
            status_mark = "[进行中]" if is_current else "[待完成]"
            
            task_data = st.get("task_data") or {}
            completion_mech = st.get("completion_mechanism", "UNKNOWN")
            
            if completion_mech == "AI_IMAGE_JUDGE":
                target_name = task_data.get("target_photo_name", "指定目标")
                raw_desc = task_data.get("target_photo_description", "请拍摄该目标")
                description = f"【拍摄目标】：{target_name} (地点: {s_loc})。 \n【背景引导】：{raw_desc}"
                description += " (⚠️系统要求：需等待 [系统通知: 图片验证通过] 才能判定完成。NPC请明确告知用户需要拍摄什么。)"
            elif completion_mech == "AI_ANSWER_CORRECT":
                description = task_data.get("ai_answer_prompt", "") or f"问题: {task_data.get('ai_question', '')}"
                description += f" (地点: {s_loc})"
            else:
                description = task_data.get("description", "请完成此步骤") + f" (地点: {s_loc})"

            reward = st.get("virtual_reward")
            reward_str = f" [奖励: {reward.get('item_semantic', '')}]" if reward else ""
            
            sub_tasks_info += f"{i+1}. {status_mark} 子任务ID: {s_id}, 名称: {st.get('game')}, 机制: {completion_mech}, 地点: {s_loc}, 描述: {description}{reward_str}\n"

            status_mark = "[当前目标]" if is_current else "[待完成]"
            
            if is_current:
                active_sub_task = st

            task_data = st.get("task_data") or {}
            completion_mech = st.get("completion_mechanism", "UNKNOWN")
            
            if completion_mech == "AI_IMAGE_JUDGE":
                target_name = task_data.get("target_photo_name", "指定目标")
                # 使用 target_photo_description 详细描述
                raw_desc = task_data.get("target_photo_description", "请拍摄该目标")
                description = f"【拍摄目标】：{target_name}。 \n【背景引导】：{raw_desc}"
                description += " (⚠️系统要求：需等待 [系统通知: 图片验证通过] 才能判定完成。NPC请明确告知用户需要拍摄什么。)"
            elif completion_mech == "AI_ANSWER_CORRECT":
                description = task_data.get("ai_answer_prompt", "") or f"问题: {task_data.get('ai_question', '')}"
            else:
                description = task_data.get("description", "请完成此步骤")

            reward = st.get("virtual_reward")
            reward_str = f" [奖励: {reward.get('item_semantic', '')}]" if reward else ""
            
            # 加入地点信息
            sub_tasks_info += f"{i+1}. {status_mark} [地点: {s_loc}] ID: {s_id}, 名称: {st.get('game')}, 机制: {completion_mech}, 描述: {description}{reward_str}\n"

    # 如果存在正在进行的子任务，将上下文聚焦到子任务
    if active_sub_task:
        # 覆盖地点
        if active_sub_task.get("location"):
            location = active_sub_task.get("location")
        # 简单追加目标描述到 objective，增强 NPC 聚焦
        sub_objective = active_sub_task.get("game", "")
        objective += f"\n【当前专注子任务】：{sub_objective}（位于 {location}）"
        
        # [FIX] Override completion_mechanism with sub-task mechanism
        s_mech = active_sub_task.get("completion_mechanism")
        if s_mech:
            if isinstance(s_mech, list) and len(s_mech) > 0:
                completion_mechanism = s_mech[0]
            elif isinstance(s_mech, str):
                completion_mechanism = s_mech

    # History
    history_text = ""
    if history:
        for h in history:
            role = h.get("role")
            content = h.get("content", "")
            if role == "user":
                history_text += f"[用户]: {content}\n"
            else:
                history_text += f"[NPC]: {content}\n"

    # Input
    safe_message = message.replace("[系统通知", "")
    user_input_with_id = f"[用户ID: {user_id}] 说: {safe_message}"
    if sub_task_id:
        user_input_with_id += f" (当前正在进行子任务: {sub_task_id})"
    
    system_notification = "无"
    if image_result:
         status = "通过" if image_result.get("success") else "失败"
         reason = image_result.get("message", "无")
         system_notification = f"[系统通知: 图片验证{status}，原因: {reason}]"

    # Time fields (Placeholders if not available)
    if time_info:
        real_time_str = time_info.get("time_str", "")
        real_date_str = time_info.get("date_str", "")
        current_solar_term = time_info.get("solar_term", "未知")
        z = time_info.get("zodiac", "未知") # Currently might only be 'Chen' or 'Si'
        sh = time_info.get("shichen", "")
        current_zodiac = f"{z} ({sh})" if sh else z
        
        # [HOTFIX] Try to calculate standard zodiac animal from current hour if possible
        try:
             # time_str usually "HH:MM" or "HH:MM:SS"
             h_part = int(real_time_str.split(":")[0])
             calculated_animal = get_zodiac_from_hour(h_part)
             current_zodiac = f"{calculated_animal}" 
        except Exception:
             pass 

    else:
        import datetime
        now = datetime.datetime.now()
        real_time_str = now.strftime("%H:%M")
        real_date_str = now.strftime("%Y-%m-%d")
        current_solar_term = "未知"
        h_part = now.hour
        calculated_animal = get_zodiac_from_hour(h_part)
        current_zodiac = calculated_animal

    # Specific fix for T06_SHADOW_CLOCK - Prepare Logic Context
    judge_context_str = str(task_knowledge)
    extra_rules_str = ""
    
    if task_id == "T06_SHADOW_CLOCK":
         judge_context_str += f"\n\n[系统强制提示]: 影子钟谜题可以通过实时时间破解。当前的正确答案是【{current_zodiac}】（或对应的生肖/地支）。"
         judge_context_str += f"\n[辅助信息]: 当前节气为【{current_solar_term}】。如果用户询问节气，请直接告诉他，这有助于寻找站位。"
         extra_rules_str = f"只要用户输入中包含正确的生肖名称（{current_zodiac}），即判定为True。忽略其他闲聊内容。"
         
    elif task_id == "T01_PROLOGUE":
         # Prologue special case: Accept confirmations
         judge_context_str += "\n\n[判定标准]: 用户表达‘同意’、‘好的’、‘出发’、‘明白了’、‘收到了’等肯定意图即视为完成。"
         extra_rules_str = "这是一个确认类任务。只要用户表达了明确的肯定/接受意图，或者表示准备好开始了，由于剧情推进需要，请判定为True。"

    elif task_id == "T07_CHESS_ACADEMY":
         # Chess Academy special case:
         extra_rules_str = "这是海龟汤猜谜。只要用户输入中包含正确谜底的核心字词（如'气生道成'或其变体），或者明确猜中了四个字的含义，即判为True。"

    # --- 1. SEPARATE JUDGMENT LOGIC ---
    internal_task_completed = False
    # AI Judgment tasks: ONLY these can update status internally
    completion_mechanisms_ai_judge = ["AI_ANSWER_CORRECT", "NPC_DIALOGUE_COMPLETE", "AI_NPC_DIALOGUE_COMPLETE", "AI_IMAGE_JUDGE"]
    # [ADD] System-Triggered Completion Mechanisms that should be respected if task_status == "completed"
    completion_mechanisms_system_judge = ["STAFF_CONFIRM", "GPS_CHECK", "COMBINE_SUCCESS"]
    
    # [FIX] Handle Image Verification implicitly
    if completion_mechanism == "AI_IMAGE_JUDGE":
        if image_result and image_result.get("success") is True:
            internal_task_completed = True
            logger.info(f"[Judge] Image verification success for task {task_id}")

    # Only run AI Judgment if mechanism is one of the above AND status is not already completed
    if task_status != "completed" and completion_mechanism in completion_mechanisms_ai_judge:
        # Skip LLM Judge for pure Image tasks (we trusted image_result above)
        if completion_mechanism == "AI_IMAGE_JUDGE":
            pass
        else:
            try:
                # JUDGE PROMPT CONFIGURATION
             judge_rules = "核心含义匹配即可。忽略标点和语气词。若是海龟汤，只要用户提到关键真相词汇即算正确。"
             if extra_rules_str:
                 judge_rules += f" 特殊规则: {extra_rules_str}"
             
             # Use a dedicated Judge Prompt
             j_prompt = judge_prompt.format(
                question=objective,
                correct_answer=judge_context_str, 
                judge_rules=judge_rules,
                user_input=safe_message
             )
             j_resp = call_api_with_retry([{"role": "user", "content": j_prompt}])
             if j_resp:
                 j_data = _simple_parse_json(j_resp)
                 logger.info(f"[Judge] Result: {j_data}") # Log the full judge result
                 if j_data.get("is_correct") is True:
                     internal_task_completed = True
                     logger.info("[Judge] User Answered Correctly.")
            except Exception as e:
                logger.error(f"[Judge] Error: {e}")

    # --- 2. PROMPT SELECTION ---
    # Determine effective status:
    # - If raw task_status is 'completed', it is completed.
    # - If AI Judge says correct (internal_task_completed), it is completed.
    # - OTHERWISE, even if user claims completion, it is 'in_progress'. 
    
    if completion_mechanism in completion_mechanisms_ai_judge:
        eff_status = "completed" if (internal_task_completed or task_status == "completed") else "in_progress"
    
    elif completion_mechanism in completion_mechanisms_system_judge:
        # [FIX] For System-Triggered tasks, if the system says 'completed', it IS completed.
        # Previously this logic was implicit, making it explicit here.
        eff_status = "completed" if (task_status == "completed") else "in_progress"
        
        # [CRITICAL] Propagate this "System Completion" to internal_task_completed 
        # so that it gets reflected in the final JSON response.
        if eff_status == "completed":
             internal_task_completed = True

    else:
        # For unknown or other tasks
        eff_status = task_status if task_status == "completed" else "in_progress"


    is_turtle_mode = ("海龟汤" in str(task_knowledge) or task_id == "T07_CHESS_ACADEMY")
    
    final_prompt_content = ""
    
    if is_turtle_mode and eff_status == "in_progress":
        # Turtle Soup Mode
        final_prompt_content = turtle_soup_prompt.format(
            npc_name=npc_name,
            npc_role=npc_role,
            style=style,
            puzzle_face=objective,
            puzzle_truth=judge_context_str,
            user_input=user_input_with_id
        )
    else:
        # Standard Interaction Mode
        final_prompt_content = npc_core_prompt.format(
            script_name=script_name,
            npc_name=npc_name,
            npc_role=npc_role,
            style=style,
            era_background=era_background,
            mechanism=mechanism,
            objective=objective,
            location=location,
            task_status=eff_status,
            selected_item=selected_item,
            hints=judge_context_str, # Passed as hints for guidance
            rag_context=rag_context if rag_context else "无",
            history=history_text,
            user_input=user_input_with_id,
            sub_tasks_info=sub_tasks_info
        )
    
    logger.info(f"[DetailedPrompt] Mode: {'Turtle' if is_turtle_mode else 'Standard'}, Status: {eff_status}")
    logger.info(f"[DetailedPrompt] RAG Context used: {bool(rag_context)}")

    messages = [{"role": "system", "content": final_prompt_content}]
    
    full_text = ""
    # Streaming call
    for part in call_api_stream_with_retry(messages):
        full_text += part
        yield ("delta", part)
        
    final_payload = _simple_parse_json(full_text)
    
    # Merge Judgment Result
    if "task_completed" not in final_payload:
        final_payload["task_completed"] = False
        
    if internal_task_completed:
        final_payload["task_completed"] = True
        
    yield ("final", final_payload)