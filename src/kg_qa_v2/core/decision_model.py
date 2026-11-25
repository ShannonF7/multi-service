"""问答 v2 的统一决策模型接口。

决策模型只负责有限集合判断，不负责数据库查询、实体召回或答案生成。
本模块同时提供：

* HostedJevClient：托管 Jev 的适配器；
* LocalDecisionClient：本地 Qwen/vLLM OpenAI-compatible 接口适配器。

本地客户端使用标准库 HTTP，避免为了决策层强制改变主项目依赖。
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, Mapping, Optional, Protocol
from urllib import request as urllib_request

from .prompts import ROUTER_SYSTEM_PROMPT


class DecisionModel(Protocol):
    """有限任务决策模型的最小协议。"""

    provider: str

    def decide(self, task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """根据任务和已整理输入返回 JSON 对象。"""


class LocalDecisionClient:
    """本地 Qwen/vLLM 的决策客户端。

    约定本地服务提供 OpenAI-compatible：
    ``POST {base_url}/chat/completions``。
    ``transport`` 仅供单元测试或部署侧注入，生产环境默认走标准库 HTTP。
    """

    provider = "local"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not base_url:
            raise ValueError("LOCAL_DECISION_BASE_URL is required")
        if not model:
            raise ValueError("LOCAL_DECISION_MODEL is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport
        self.stream_router = os.getenv("LOCAL_DECISION_STREAM_ROUTER", "1").strip().lower() not in {"0", "false", "no"}
        self.last_timing: dict[str, int] = {}
        self.model_backed = True

    def decide(self, task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        if task == "router":
            self.last_timing = {}
        body = self._build_request(task, payload)
        request_started = time.perf_counter()
        attempts = 0
        if self.transport:
            attempts += 1
            raw = self.transport(body, self.base_url, self.api_key, self.timeout)
        elif task == "router" and self.stream_router:
            attempts += 1
            try:
                raw = self._request_stream(body)
            except Exception:
                # 兼容不支持 SSE 的 OpenAI-compatible 服务，只回退一次非流式请求。
                attempts += 1
                raw = self._request(body)
        else:
            attempts += 1
            raw = self._request(body)
        request_finished = time.perf_counter()
        parse_started = request_finished
        content = _extract_content(raw)
        result = _parse_json_object(content)
        if task == "router":
            self.last_timing["router_http_ms"] = int((request_finished - request_started) * 1000)
            self.last_timing["router_parse_ms"] = int((time.perf_counter() - parse_started) * 1000)
            self.last_timing["router_attempt_count"] = attempts
            self.last_timing["router_model_calls"] = attempts
            self.last_timing.setdefault("router_generation_ms", int((request_finished - request_started) * 1000))
            self.last_timing.setdefault("router_total_ms", int((time.perf_counter() - started) * 1000))
            return _normalize_router(result)
        return result

    def _build_request(self, task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": _task_instruction(task),
                },
                {
                    "role": "user",
                    "content": json.dumps(dict(payload), ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        if task == "router":
            body["max_tokens"] = _bounded_env_int("LOCAL_ROUTER_MAX_TOKENS", 64, 32, 128)
        elif task == "entity_alignment":
            body["max_tokens"] = _bounded_env_int("LOCAL_ALIGNMENT_MAX_TOKENS", 96, 32, 192)
        elif task == "citation_validation":
            body["max_tokens"] = _bounded_env_int("LOCAL_CITATION_MAX_TOKENS", 128, 32, 256)
        return body

    def _request(self, body: Dict[str, Any]) -> Any:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        req = urllib_request.Request(endpoint, data=data, headers=headers, method="POST")
        with urllib_request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_stream(self, body: Dict[str, Any]) -> Any:
        started = time.perf_counter()
        stream_body = dict(body)
        stream_body["stream"] = True
        data = json.dumps(stream_body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        req = urllib_request.Request(endpoint, data=data, headers=headers, method="POST")
        chunks: list[str] = []
        first_chunk_at: float | None = None
        with urllib_request.urlopen(req, timeout=self.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                choices = value.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    chunks.append(str(content))
        finished = time.perf_counter()
        if first_chunk_at is not None:
            ttft_ms = int((first_chunk_at - started) * 1000)
            total_ms = int((finished - started) * 1000)
            self.last_timing = {
                "router_ttft_ms": ttft_ms,
                "router_generation_ms": max(0, total_ms - ttft_ms),
            }
        else:
            self.last_timing = {"router_generation_ms": int((finished - started) * 1000)}
        if not chunks:
            raise RuntimeError("Router SSE 未返回可解析内容")
        return {"choices": [{"message": {"content": "".join(chunks)}}]}


class TransformersDecisionClient:
    """直接在当前 Python 进程中加载 Transformers 因果语言模型。

    这是本地模型服务之外的第二条路径。模型采用懒加载，只有第一次决策时
    才读取权重；测试和部署侧可以注入 ``generator``，避免测试下载模型。
    """

    provider = "local"

    def __init__(
        self,
        *,
        model_path: str,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        model: Any = None,
        tokenizer: Any = None,
        generator: Optional[Callable[[str], str]] = None,
    ) -> None:
        if not model_path:
            raise ValueError("LOCAL_DECISION_MODEL_PATH is required")
        self.model_path = model_path
        self.max_new_tokens = max(32, min(int(max_new_tokens), 1024))
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._model = model
        self._tokenizer = tokenizer
        self.generator = generator

    def decide(self, task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(task, payload)
        content = self.generator(prompt) if self.generator else self._generate(prompt)
        result = _parse_json_object(content)
        return _normalize_router(result) if task == "router" else result

    @staticmethod
    def _build_prompt(task: str, payload: Mapping[str, Any]) -> str:
        messages = [
            {"role": "system", "content": _task_instruction(task)},
            {"role": "user", "content": json.dumps(dict(payload), ensure_ascii=False)},
        ]
        return json.dumps(messages, ensure_ascii=False)

    def _generate(self, prompt: str) -> str:
        model, tokenizer = self._load()
        messages = json.loads(prompt)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("本地 tokenizer 不支持 apply_chat_template")
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        torch = self._torch()
        with torch.no_grad():
            output = model.generate(
                **self._move_inputs(encoded, model),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        input_ids = encoded["input_ids"]
        generated = output[0][input_ids.shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("未安装 transformers，无法使用进程内本地决策模型") from exc
        self._tokenizer = self._tokenizer or AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = self._model or AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype="auto",
            trust_remote_code=self.trust_remote_code,
        )
        return self._model, self._tokenizer

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("未安装 torch，无法使用进程内本地决策模型") from exc
        return torch

    @staticmethod
    def _move_inputs(encoded: Mapping[str, Any], model: Any) -> Dict[str, Any]:
        device = getattr(model, "device", None)
        if device is None:
            return dict(encoded)
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }


class HostedJevClient:
    """把已有 TypeSafe Jev 客户端暴露为统一 DecisionModel。"""

    provider = "jev"

    def __init__(self, client: Any) -> None:
        self.client = client

    def decide(self, task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if hasattr(self.client, "decide"):
            return dict(self.client.decide(task, dict(payload)))
        if task != "router":
            raise NotImplementedError(f"当前 Jev 适配器未实现任务：{task}")
        return dict(self.client(state=dict(payload)))


class CascadingDecisionClient:
    """Router 级联：轻量模型低置信度时才调用 fallback。"""

    provider = "local"

    def __init__(self, primary: DecisionModel, fallback: Optional[DecisionModel], *, threshold: float = 0.82) -> None:
        self.primary = primary
        self.fallback = fallback
        self.threshold = max(0.5, min(float(threshold), 0.99))
        # 级联客户端内部仍然保留生产本地 Router 的确定性 Fast Path。
        self.supports_fast_path = True
        self.last_timing: dict[str, int] = {}
        self.last_meta: dict[str, str] = {}

    def decide(self, task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if task != "router" or self.primary is None:
            return self._fallback(task, payload)
        primary_calls = 0
        fallback_reason = "primary_low_confidence"
        try:
            if getattr(self.primary, "model_backed", True):
                primary_calls = 1
            result = _normalize_router(self.primary.decide(task, payload))
            if result.get("intent") != "UNKNOWN" and float(result.get("confidence") or 0.0) >= self.threshold:
                self.last_timing = dict(getattr(self.primary, "last_timing", {}) or {})
                self.last_timing["router_model_calls"] = max(primary_calls, int(self.last_timing.get("router_model_calls") or 0))
                self.last_timing["router_attempt_count"] = self.last_timing["router_model_calls"]
                self.last_meta = {"router_source": "primary"}
                return result
        except Exception:
            fallback_reason = "primary_error"
        result = self._fallback(task, payload)
        fallback_calls = int(self.last_timing.get("router_model_calls") or 0)
        if getattr(self.fallback, "model_backed", True) and fallback_calls == 0:
            fallback_calls = 1
        self.last_timing["router_model_calls"] = primary_calls + fallback_calls
        self.last_timing["router_attempt_count"] = self.last_timing["router_model_calls"]
        self.last_meta = {"router_source": "fallback", "router_fallback_reason": fallback_reason}
        return result

    def _fallback(self, task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if self.fallback is None:
            raise RuntimeError("Router 级联没有可用的 fallback 决策模型")
        result = self.fallback.decide(task, payload)
        self.last_timing = dict(getattr(self.fallback, "last_timing", {}) or {})
        return result


class LightweightRouterClient:
    """无模型的轻量 Router，作为 7B 之前的确定性决策层。"""

    provider = "local"
    model_backed = False

    def decide(self, task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if task != "router":
            return {"intent": "UNKNOWN", "confidence": 0.0}
        question = str(payload.get("question") or "").strip()
        compact = re.sub(r"[\s，。！？?；;、]+", "", question).casefold()
        if any(marker in question for marker in ("心情怎么样", "心情如何", "心情好吗")):
            return {"intent": "CASUAL", "context_dependency": "NONE", "confidence": 0.90}
        if re.search(r"^(?:我|今天|最近)?(?:好|有点|太|真的)?(?:累|困|疲惫|难过|开心|郁闷|烦|焦虑|紧张)(?:啊|呀|了|呢)?$", compact):
            return {"intent": "CASUAL", "context_dependency": "NONE", "confidence": 0.96}
        if any(marker in compact for marker in ("讲个笑话", "说个笑话", "讲个故事", "陪我聊聊", "聊聊天", "逗我一下", "有点无聊")):
            return {"intent": "CASUAL", "context_dependency": "NONE", "confidence": 0.96}
        if any(marker in question for marker in ("图片", "照片", "图像", "配图", "长什么样")):
            return {"intent": "IMAGE", "context_dependency": "IMAGE", "confidence": 0.90}
        if any(marker in question for marker in ("怎么去", "如何到达", "路线")) or re.search(r"从.+到", question):
            return {"intent": "PATH", "context_dependency": "NONE", "confidence": 0.86}
        return {"intent": "UNKNOWN", "context_dependency": "NONE", "confidence": 0.0}


def build_decision_model_from_env() -> Optional[DecisionModel]:
    """按环境变量选择本地或托管决策模型。

    ``DECISION_MODEL_PROVIDER=local``：必须配置本地地址和模型。
    ``hosted_jev``：使用已有 Jev API。
    ``auto``：配置本地模型时优先本地，否则尝试 Jev。
    """

    provider = os.getenv("DECISION_MODEL_PROVIDER", "auto").strip().lower()
    local_backend = os.getenv("LOCAL_DECISION_BACKEND", "http").strip().lower()
    local_model_path = os.getenv("LOCAL_DECISION_MODEL_PATH")
    local_base_url = os.getenv("LOCAL_DECISION_BASE_URL") or os.getenv("VLLM_BASE_URL")
    local_model = os.getenv("LOCAL_DECISION_MODEL") or os.getenv("VLLM_MODEL")
    fallback: Optional[DecisionModel] = None
    if provider in {"local", "auto"} and local_backend in {"transformers", "in_process"} and local_model_path:
        fallback = TransformersDecisionClient(
            model_path=local_model_path,
            max_new_tokens=int(os.getenv("LOCAL_DECISION_MAX_NEW_TOKENS", "256")),
            device_map=os.getenv("LOCAL_DECISION_DEVICE_MAP", "auto"),
        )
    elif provider in {"local", "auto"} and local_base_url and local_model:
        fallback = LocalDecisionClient(
            base_url=local_base_url,
            model=local_model,
            api_key=os.getenv("LOCAL_DECISION_API_KEY") or os.getenv("VLLM_API_KEY"),
            timeout=_env_float("LOCAL_DECISION_TIMEOUT", 60.0),
        )
    elif provider in {"jev", "hosted_jev", "auto"}:
        from ..adapters.jev_client import build_jev_client_from_env

        jev = build_jev_client_from_env()
        fallback = HostedJevClient(jev) if jev is not None else None
    if provider == "local" and fallback is None:
        raise ValueError("DECISION_MODEL_PROVIDER=local 时必须配置本地地址和模型")

    router_base = os.getenv("LOCAL_ROUTER_BASE_URL") or os.getenv("ROUTER_BASE_URL")
    router_model = os.getenv("LOCAL_ROUTER_MODEL") or os.getenv("ROUTER_MODEL")
    if router_base and router_model:
        primary = LocalDecisionClient(
            base_url=router_base,
            model=router_model,
            api_key=os.getenv("LOCAL_ROUTER_API_KEY") or os.getenv("ROUTER_API_KEY"),
            timeout=_env_float("LOCAL_ROUTER_TIMEOUT", 8.0),
        )
        return CascadingDecisionClient(
            primary,
            fallback,
            threshold=float(os.getenv("LOCAL_ROUTER_CONFIDENCE_THRESHOLD", "0.82")),
        )
    if provider == "auto" and fallback is not None:
        return CascadingDecisionClient(
            LightweightRouterClient(),
            fallback,
            threshold=float(os.getenv("LOCAL_ROUTER_CONFIDENCE_THRESHOLD", "0.82")),
        )
    return fallback


def _task_instruction(task: str) -> str:
    instructions = {
        "router": ROUTER_SYSTEM_PROMPT,
        "entity_alignment": (
            "你是候选对象对齐器。只能从 candidates 中选择，不能创造候选，不能跨当前领域选择。\n"
            "必须返回 JSON：selected_node_id、confidence、reason；无法确定时 selected_node_id=UNKNOWN。"
        ),
        "citation_validation": (
            "你是证据引用校验器。判断 answer_sentence 是否被 evidence 支持。\n"
            "必须返回 JSON：status 只能是 ENTAILED、PARTIAL、NOT_ENTAILED，另有 confidence、reason。"
        ),
    }
    return instructions.get(task, "只返回合法 JSON 对象，不要输出 Markdown 或额外解释。")


def _normalize_router(value: Dict[str, Any]) -> Dict[str, Any]:
    intents = {"CASUAL", "DIRECT_FACT", "LOCATION", "HIERARCHY", "PATH", "IMAGE", "COMPARE", "EXPLAIN", "LIST", "UNKNOWN"}
    intent = str(value.get("intent") or "UNKNOWN").upper()
    if intent not in intents:
        intent = "UNKNOWN"
    dependency = str(value.get("context_dependency") or "").upper()
    if dependency not in {"NONE", "SESSION", "LOCATION", "PENDING_CLARIFICATION", "IMAGE"}:
        dependency = "NONE"
    return {
        "intent": intent,
        "context_dependency": "NONE" if intent == "CASUAL" else dependency,
        "confidence": _clamp(value.get("confidence", 0.0)),
    }


def _extract_content(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        if isinstance(raw, dict) and "choices" in raw:
            choices = raw.get("choices") or []
            if choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else getattr(choices[0], "message", {})
                return message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        return raw
    return getattr(raw, "content", raw)


def _parse_json_object(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("本地决策模型未返回合法 JSON")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("本地决策模型返回的 JSON 顶层必须是对象")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 0.7
    return str(value or "").strip().lower() in {"true", "yes", "1", "是", "需要"}


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))
