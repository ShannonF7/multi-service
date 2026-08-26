from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from src.llm.utils import call_api_with_retry
from src.llm.prompts.prompts import EVAL_SYSTEM_PROMPT

# =========================
# Configuration & Paths
# =========================

DEFAULT_DECISION_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "decision.jsonl")
)
DEFAULT_TRACE_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "llm_trace.jsonl")
)
DEFAULT_SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "prompts", "scripts_template.json")
)

# =========================
# State Machine
# =========================

class TaskState(str, Enum):
    IDLE = "IDLE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class StateEvent(str, Enum):
    START = "START"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    RESET = "RESET"

STATE_TRANSITIONS = {
    TaskState.IDLE: {
        StateEvent.START: TaskState.IN_PROGRESS,
        StateEvent.COMPLETE: TaskState.COMPLETED, 
        StateEvent.FAIL: TaskState.FAILED, 
    },
    TaskState.IN_PROGRESS: {
        StateEvent.COMPLETE: TaskState.COMPLETED,
        StateEvent.FAIL: TaskState.FAILED,
    },
    TaskState.FAILED: {
        StateEvent.RESET: TaskState.IDLE,
    },
    TaskState.COMPLETED: {
        StateEvent.RESET: TaskState.IDLE, 
    },
}

def apply_state_event(prev: TaskState, event: StateEvent) -> TaskState:
    return STATE_TRANSITIONS.get(prev, {}).get(event, prev)

# =========================
# Data Models
# =========================

@dataclass
class DialogueEvalResult:
    task_id: str
    signals: Dict[str, Any]
    score: float
    refusal: bool
    is_task_related: bool = True  # 是否与当前任务目标相关

@dataclass
class RuleDecision:
    task_completed: bool
    suggested_event: StateEvent
    assessment: str
    message: str
    score: float
    is_bypassed: bool = False  # 是否跳过了规则引擎处理过程

@dataclass
class PipelineOutput:
    task_id: str
    eval_result: DialogueEvalResult
    rule: RuleDecision
    prev_state: TaskState
    next_state: TaskState

# =========================
# Config Loader
# =========================

def load_task_config(task_id: str, script_path: str = None) -> Dict[str, Any]:
    path = script_path or DEFAULT_SCRIPT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config path {path} not found")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for t in data.get("tasks", []):
        if t.get("task_id") == task_id:
            return t

    raise ValueError(f"Task {task_id} not found in {path}")

# =========================
# LLM Signal Extraction
# =========================

def extract_signals_with_llm(
    task_id: str,
    user_text: str,
    criteria_list: List[Dict[str, Any]],
    trace_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """使用 LLM 提取判定项命中度和拒绝标志。"""
    
    # 构建判定项描述和 JSON 模板
    criteria_desc = "\n".join([f"- {c['id']}: {c['type']}: {c['desc']}" for c in criteria_list])
    criteria_json_template = ", ".join([f'"{c["id"]}": 0.0' for c in criteria_list])

    user_prompt = f"""
任务ID: {task_id}
用户输入: "{user_text}"

待抽取的判定项（严格遵循指代一致性）：
{criteria_desc}

重要指令：
1. 如果用户仅说“拿到了”、“好了”等模糊动词，但没有提到具体的判定项名称或其核心关键词，请将相关项判为 0.0。
2. 禁止脑补用户指代。例如，如果任务包含“门票”和“指引”，用户说“拿到了”，既不能判“门票”1.0，也不能判“指引”1.0。

输出 JSON 格式示例：
{{
  "hits": {{
    {criteria_json_template}
  }},
  "refusal": false
}}
"""

    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_api_with_retry(messages)
    
    try:
        # 清理响应内容中的 Markdown 块
        clean_res = response.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_res)
    except Exception:
        data = {"hits": {}, "refusal": False}

    if trace_log_path:
       os.makedirs(os.path.dirname(trace_log_path), exist_ok=True)
       with open(trace_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "task_id": task_id,
                "user_text": user_text,
                "llm_output": data
            }, ensure_ascii=False) + "\n")

    return data

# =========================
# Core Logic: Scoring & Rules
# =========================

def aggregate_completion_score(
    criteria: List[Dict[str, Any]],
    hits: Dict[str, float],
) -> float:
    """结合权重计算总命中得分（仅计入 fact 类型，用于判定是否完成）。"""
    score = 0.0
    for c in criteria:
        # 只有 fact 类型计入任务完成分，degree 类型（如语气）仅用于规则细化
        if c.get("type", "fact") != "fact":
            continue
        cid = c["id"]
        weight = c.get("weight", 0.0)
        score += weight * hits.get(cid, 0.0)
    return round(score, 3) # 保留三位小数

def decide_by_rules(
    *,
    score: float,
    pass_threshold: float,
    refusal: bool,
    hits: Dict[str, float],
    triggers: List[Dict[str, Any]],
) -> RuleDecision:
    """
    根据得分、拒绝情况以及特定触发器决定任务状态。
    """
    
    # 1. 逻辑判定引擎 (priority + conditions)
    new_triggers = [t for t in triggers if "conditions" in t]
    if new_triggers:
        # 按优先级从高到低排序
        sorted_triggers = sorted(new_triggers, key=lambda x: x.get("priority", 0), reverse=True)
        for t in sorted_triggers:
            match = True
            for cond in t.get("conditions", []):
                c_type = cond.get("type")
                c_id = cond.get("id")
                op = cond.get("operator")
                target_val = cond.get("value")

                # 获取当前值 (根据类型灵活获取)
                if c_type == "refusal":
                    actual = refusal
                elif c_type == "completion":
                    actual = "completed" if score >= pass_threshold else "not_completed"
                elif c_type == "score":
                    actual = score
                else:
                    actual = hits.get(c_id, 0.0)

                # 条件比较逻辑
                if op == "==":
                    if actual != target_val: match = False; break
                elif op == ">=":
                    if actual < target_val: match = False; break
                elif op == "<":
                    if actual >= target_val: match = False; break
                elif op == ">":
                    if actual <= target_val: match = False; break
                elif op == "<=":
                    if actual > target_val: match = False; break
                else:
                    match = False; break
            
            if match:
                return _create_decision(t["action"], score)

   # 2. 兜底逻辑
    is_passed = score >= pass_threshold
    return RuleDecision(
        task_completed=is_passed,
        suggested_event=StateEvent.COMPLETE if is_passed else StateEvent.START,
        assessment="success" if is_passed else "info",
        message="任务完成" if is_passed else "继续努力",
        score=score
    )

def _create_decision(action: Dict[str, Any], score: float) -> RuleDecision:
    """根据动作配置生成决策对象。"""
    assessment = action.get("assessment", "info")
    
    # 状态机事件映射
    event_map = {
        "fail": StateEvent.FAIL,
        "success": StateEvent.COMPLETE,
        "warning": StateEvent.COMPLETE,
        "info": StateEvent.START
    }
    
    return RuleDecision(
        task_completed=assessment in ("success", "warning"),
        suggested_event=event_map.get(assessment, StateEvent.START),
        assessment=assessment,
        message=action.get("content", ""),
        score=score
    )

# =========================
# Pipeline Entry
# =========================

def run_task_pipeline(
    *,
    task_id: str,
    user_text: str,
    prev_state: TaskState,
    task_config: Dict[str, Any] = None,
) -> PipelineOutput:
    """三段式 Agent 管道：识别 -> 过滤相关性 -> 自动评分 -> 规则更新。"""

    # 1. 加载配置 (Backwards compatibility or passed config)
    if task_config:
        task = task_config
    else:
        task = load_task_config(task_id)
        
    criteria = task.get("completion_criteria", {}).get("criteria", [])
    pass_threshold = task.get("completion_criteria", {}).get("pass_threshold", 0.7)

    # 2. LLM 信号提取 (识别意图命中的事实)
    signals = extract_signals_with_llm(
        task_id=task_id,
        user_text=user_text,
        criteria_list=criteria,
        trace_log_path=DEFAULT_TRACE_LOG_PATH,
    )

    hits = signals.get("hits", {})
    refusal = signals.get("refusal", False)

    # 3. 相关性自检 (Agent 特有的意图过滤逻辑)
    # 逻辑：仅当存在事实项（fact）命中或明确拒绝时，才视为任务相关对话。
    # 纯粹的语气好（tone）或无关闲聊不应触发规则引擎。
    fact_ids = {c['id'] for c in criteria if c.get("type", "fact") == "fact"}
    total_fact_hit = sum(hits.get(fid, 0.0) for fid in fact_ids)
    is_task_related = (total_fact_hit > 0) or (refusal is True)

    if not is_task_related:
        # 构建 Bypass 结果：不改变状态机，不提供评估信息
        eval_res = DialogueEvalResult(
            task_id=task_id,
            signals=signals,
            score=0.0,
            refusal=False,
            is_task_related=False
        )
        rule = RuleDecision(
            task_completed=False,
            suggested_event=StateEvent.START, # 保持在起始/进行中状态
            assessment="none",                # 特殊评估：无目标命中
            message="",                       # 不生成规则引导话术
            score=0.0,
            is_bypassed=True
        )
        return PipelineOutput(
            task_id=task_id,
            eval_result=eval_res,
            rule=rule,
            prev_state=prev_state,
            next_state=prev_state             # 状态保持不变
        )

    # 4. 如果判定相关，则计算聚合分数并驱动规则引擎
    score = aggregate_completion_score(criteria, hits)
    rule = decide_by_rules(
        score=score,
        pass_threshold=pass_threshold,
        refusal=refusal,
        hits=hits,
        triggers=task.get("triggers", []),
    )

    # 5. 更新状态并输出完整结果
    next_state = apply_state_event(prev_state, rule.suggested_event)
    output = PipelineOutput(
        task_id=task_id,
        eval_result=DialogueEvalResult(
            task_id=task_id,
            signals=signals,
            score=score,
            refusal=refusal,
            is_task_related=True
        ),
        rule=rule,
        prev_state=prev_state,
        next_state=next_state,
    )

    _append_decision_log(output, user_text)
    return output

def _append_decision_log(output: PipelineOutput, user_text: str):
    """记录决策审计日志。"""
    os.makedirs(os.path.dirname(DEFAULT_DECISION_LOG_PATH), exist_ok=True)
    with open(DEFAULT_DECISION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": time.time(),
             "task_id": output.task_id,
            "input": user_text,
            "score": output.rule.score,
            "assessment": output.rule.assessment,
            "next_state": output.next_state,
        }, ensure_ascii=False) + "\n")
