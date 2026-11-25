"""新 AI 的隔离适配器、默认答案生成和证据校验。"""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Protocol

from .schemas import Citation, DomainContext, EvidenceItem, InputAnalysis, LinkedEntity, QueryPlan


class QAAnswerGenerator(Protocol):
    def generate(
        self,
        *,
        question: str,
        context: DomainContext,
        analysis: InputAnalysis,
        entities: List[LinkedEntity],
        plan: QueryPlan,
        evidence: List[EvidenceItem],
    ) -> Dict[str, Any]: ...
    def generate_stream(
        self,
        *,
        question: str,
        context: DomainContext,
        analysis: InputAnalysis,
        entities: List[LinkedEntity],
        plan: QueryPlan,
        evidence: List[EvidenceItem],
        on_delta: Callable[[str], None],
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...


def build_grounded_prompt(*, question, context, analysis, entities, plan, evidence, conversation_context=None) -> str:
    """构造受已发布领域证据约束的提示词；明确禁止 Web Search。"""
    entity_text = "、".join(item.name or item.mention for item in entities if item.status == "LINKED") or "无"
    evidence_text = "\n".join(f"[{item.evidence_id}] {item.content}" for item in evidence) or "无可用证据"
    conversation_text = conversation_context or {}
    return (
        "你是领域知识问答助手。只能依据给定领域和证据回答。\n"
        "禁止联网搜索，禁止补充证据中没有出现的事实。\n"
        "如果证据不足，请明确说明无法确认。回答中的每条事实句末都使用 [evidence_id] 标注来源；不要单独输出引用列表。不要输出 node_id、节点编号或“实体”等技术术语。回答使用自然的中文分段，多个事实分成简短段落，列表内容逐项换行。对于“在哪里、在哪、位于”这类问题，只回答目标位置本身的地址、坐标或所属区域，不要罗列其下属设施，除非问题明确要求列出周边或内部位置。\n"
        f"领域：{context.name or context.domain_id}\n"
        f"问题：{question}\n"
        f"意图：{analysis.knowledge_intents}\n"
        f"相关位置或对象：{entity_text}\n"
        f"计划：{[item.type for item in plan.steps]}\n"
        f"历史会话（仅用于保持上下文，不是证据）：{conversation_text}\n"
        f"证据：\n{evidence_text}"
    )


class CallableAIAnswerGenerator:
    def __init__(self, client: Callable[..., Any]) -> None:
        self.client = client

    def generate(self, *, question, context, analysis, entities, plan, evidence, conversation_context=None):
        prompt = build_grounded_prompt(
            question=question, context=context, analysis=analysis,
            entities=entities, plan=plan, evidence=evidence, conversation_context=conversation_context,
        )
        result = self.client(
            prompt=prompt,
            evidence=[item.model_dump() if hasattr(item, "model_dump") else item.dict() for item in evidence],
            web_search=False,
        )
        if isinstance(result, str):
            return {"answer": result, "citations": []}
        return dict(result or {})

    def generate_stream(self, *, question, context, analysis, entities, plan, evidence, on_delta, conversation_context=None):
        result = self.generate(
            question=question, context=context, analysis=analysis, entities=entities,
            plan=plan, evidence=evidence, conversation_context=conversation_context,
        )
        answer = str(result.get("answer") or "")
        if answer:
            on_delta(answer)
        return result


class DashScopeQAClient:
    """KG-QA 专用 AI 客户端；默认关闭 DashScope 的联网搜索能力。"""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("KG_QA_LLM_MODEL", "qwen-plus")

    def __call__(self, *, prompt: str, evidence: List[Dict[str, Any]], web_search: bool = False) -> Dict[str, Any]:
        if web_search:
            raise ValueError("KG-QA answer client does not allow web search")
        from dashscope import Generation
        from src.core.config import settings

        api_key = getattr(settings, "LLM_API_KEY", None) or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("KG-QA 未配置 LLM_API_KEY")
        response = Generation.call(
            api_key=api_key,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你只能依据用户提供的领域证据回答，禁止联网搜索。请输出 JSON；"
                        "事实句末必须保留 [evidence_id] 引用标记，并使用自然中文分段；不要输出 node_id、节点编号或“实体”等技术术语："
                        '{"answer":"...","citations":[{"evidence_id":"..."}]}'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            enable_search=False,
        )
        content = self._extract_content(response)
        parsed = self._parse_json(content)
        return parsed if parsed is not None else {"answer": content, "citations": []}

    def generate_stream(self, *, question, context, analysis, entities, plan, evidence, on_delta, conversation_context=None):
        if os.getenv("KG_QA_LLM_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
            return {"answer": "", "citations": []}
        prompt = build_grounded_prompt(
            question=question, context=context, analysis=analysis,
            entities=entities, plan=plan, evidence=evidence,
            conversation_context=conversation_context,
        )
        from dashscope import Generation
        from src.core.config import settings
        api_key = getattr(settings, "LLM_API_KEY", None) or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("KG-QA 未配置 LLM_API_KEY")
        response = Generation.call(
            api_key=api_key,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你只能依据用户提供的领域证据回答，禁止联网搜索。"
                        "只输出答案正文，不要输出 JSON、Markdown 引用列表、node_id、节点编号或“实体”等技术术语。"
                        "每条事实句末必须保留对应的 [evidence_id] 引用标记，使用自然中文分段。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            enable_search=False,
            stream=True,
            incremental_output=True,
        )
        chunks = response if hasattr(response, "__iter__") and not hasattr(response, "output") else [response]
        parts: list[str] = []
        stream_buffer = ""
        pending_tail = ""
        streaming_plain_text = False

        def emit_plain(value: str) -> None:
            nonlocal pending_tail
            combined = pending_tail + value
            marker = '{"answer"'
            marker_pos = combined.find(marker)
            if marker_pos >= 0:
                prefix = combined[:marker_pos]
                if prefix:
                    on_delta(html.unescape(prefix))
                pending_tail = combined[marker_pos:]
                return
            hold = 0
            for size in range(min(len(combined), len(marker) - 1), 0, -1):
                if combined.endswith(marker[:size]):
                    hold = size
                    break
            if hold:
                on_delta(html.unescape(combined[:-hold]))
                pending_tail = combined[-hold:]
            else:
                on_delta(html.unescape(combined))
                pending_tail = ""

        for chunk in chunks:
            content = self._extract_content(chunk)
            if not content:
                continue
            parts.append(content)
            stream_buffer += content
            if not streaming_plain_text:
                probe = stream_buffer.lstrip()
                if probe.startswith("{") or probe.startswith("```"):
                    continue
                streaming_plain_text = True
                emit_plain(stream_buffer)
            else:
                emit_plain(content)
        raw_answer = html.unescape("".join(parts)).strip()
        parsed = self._parse_json(raw_answer)
        embedded = _extract_embedded_answer_payload(raw_answer) if parsed is None else None
        if parsed is not None and "answer" in parsed:
            answer = html.unescape(str(parsed.get("answer") or "")).strip()
            if not streaming_plain_text:
                on_delta(answer)
            return {"answer": answer, "citations": parsed.get("citations") or []}
        if embedded is not None:
            _, payload = embedded
            answer = html.unescape(str(payload.get("answer") or "")).strip()
            if not streaming_plain_text:
                on_delta(answer)
            return {"answer": answer, "citations": payload.get("citations") or []}
        if pending_tail:
            on_delta(html.unescape(pending_tail))
        if not streaming_plain_text and raw_answer:
            on_delta(raw_answer)
        return {"answer": raw_answer, "citations": []}

    @staticmethod
    def _extract_content(response: Any) -> str:
        output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
        choices = output.get("choices") if isinstance(output, dict) else getattr(output, "choices", None)
        first = (choices or [None])[0]
        message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content).strip()
        return str(content or "").strip()

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict[str, Any]]:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None


class CasualChatGenerator:
    """闲聊模式：不访问图谱、不生成引用，只提供自然对话。"""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("KG_QA_LLM_MODEL", "qwen-plus")

    def generate(self, message: str, *, conversation_context=None) -> str:
        fallback = "我在呢。想聊天可以直接和我说，也可以问我当前领域的知识。"
        if os.getenv("KG_QA_LLM_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
            return fallback
        try:
            from dashscope import Generation
            from src.core.config import settings

            api_key = getattr(settings, "LLM_API_KEY", None) or os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                return fallback
            response = Generation.call(
                api_key=api_key,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是小V，当前请求是轻松闲聊。请自然、友好、简短地回复；"
                            "不要检索图谱，不要联网，不要添加引用，不要臆造专业事实。"
                        ),
                    },
                    *((conversation_context or {}).get("messages") or [])[-8:],
                    {"role": "user", "content": message},
                ],
                result_format="message",
                enable_search=False,
            )
            content = DashScopeQAClient._extract_content(response)
            return content or fallback
        except Exception:
            return fallback


    def generate_stream(self, message: str, *, conversation_context=None, on_delta=None) -> str:
        fallback = "我在呢。想聊天可以直接和我说，也可以问我当前领域的知识。"
        if os.getenv("KG_QA_LLM_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
            if on_delta:
                on_delta(fallback)
            return fallback
        try:
            from dashscope import Generation
            from src.core.config import settings
            api_key = getattr(settings, "LLM_API_KEY", None) or os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                if on_delta:
                    on_delta(fallback)
                return fallback
            history = (conversation_context or {}).get("messages") or []
            messages = [
                {
                    "role": "system",
                    "content": "你是小V，当前请求是轻松闲聊。自然、友好、简短地回复；不要检索图谱、联网或添加引用。",
                },
                *history[-8:],
                {"role": "user", "content": message},
            ]
            response = Generation.call(
                api_key=api_key, model=self.model, messages=messages,
                result_format="message", enable_search=False,
                stream=True, incremental_output=True,
            )
            chunks = response if hasattr(response, "__iter__") and not hasattr(response, "output") else [response]
            parts = []
            for chunk in chunks:
                content = DashScopeQAClient._extract_content(chunk)
                if content:
                    parts.append(content)
                    if on_delta:
                        on_delta(content)
            return "".join(parts) or fallback
        except Exception:
            if on_delta:
                on_delta(fallback)
            return fallback


class FallbackAnswerGenerator:
    def generate(self, *, question, context, analysis, entities, plan, evidence, conversation_context=None):
        if not evidence:
            return {"answer": "当前领域资料中没有找到足够证据回答这个问题。", "citations": []}
        answer = "；".join(item.content for item in evidence[:3])
        return {
            "answer": answer,
            "citations": [{"evidence_id": item.evidence_id} for item in evidence[:3]],
        }


    def generate_stream(self, *, question, context, analysis, entities, plan, evidence, on_delta, conversation_context=None):
        result = self.generate(
            question=question, context=context, analysis=analysis, entities=entities,
            plan=plan, evidence=evidence, conversation_context=conversation_context,
        )
        answer = str(result.get("answer") or "")
        if answer:
            on_delta(answer)
        return result


class SafeQAAnswerGenerator:
    def __init__(self, primary: QAAnswerGenerator, fallback: QAAnswerGenerator) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(self, **kwargs: Any) -> Dict[str, Any]:
        try:
            return self.primary.generate(**kwargs)
        except Exception as exc:
            result = dict(self.fallback.generate(**kwargs))
            result.setdefault("warnings", []).append(f"新 AI 调用失败，已使用证据降级答案：{exc}")
            return result

    def generate_stream(self, *, on_delta, **kwargs: Any) -> Dict[str, Any]:
        try:
            stream_method = getattr(self.primary, "generate_stream", None)
            if stream_method is None:
                raise RuntimeError("主答案生成器未提供流式接口")
            return dict(stream_method(on_delta=on_delta, **kwargs) or {})
        except Exception as exc:
            result = dict(self.fallback.generate(**kwargs))
            answer = str(result.get("answer") or "")
            if answer:
                on_delta(answer)
            result.setdefault("warnings", []).append(f"流式 AI 调用失败，已使用证据降级答案：{exc}")
            return result


def build_default_answer_generator() -> QAAnswerGenerator:
    fallback = FallbackAnswerGenerator()
    if os.getenv("KG_QA_LLM_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
        return fallback
    return SafeQAAnswerGenerator(CallableAIAnswerGenerator(DashScopeQAClient()), fallback)


def _extract_embedded_answer_payload(value: str) -> tuple[int, Dict[str, Any]] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{\s*"answer"\s*:', value):
        try:
            payload, _ = decoder.raw_decode(value[match.start():])
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and "answer" in payload:
            return match.start(), payload
    return None


def _normalize_generated_payload(generated: Dict[str, Any]) -> Dict[str, Any]:
    """清理模型答案，避免 中文包装、HTML 实体和引用格式泄漏到前端。"""
    result = dict(generated or {})
    answer = html.unescape(str(result.get("answer") or result.get("text") or "")).strip()
    parsed = DashScopeQAClient._parse_json(answer)
    prefix = ""
    if parsed is None:
        embedded = _extract_embedded_answer_payload(answer)
        if embedded is not None:
            position, parsed = embedded
            prefix = answer[:position].rstrip()
    if parsed is not None and "answer" in parsed:
        parsed_answer = html.unescape(str(parsed.get("answer") or "")).strip()
        answer = prefix or parsed_answer
        if not result.get("citations"):
            result["citations"] = parsed.get("citations") or []
    answer = re.sub(r"\*\*\s*(\[[^\[\]]+\])\s*\*\*", r"\1", answer)
    answer = re.sub(r"__\s*(\[[^\[\]]+\])\s*__", r"\1", answer)
    result["answer"] = answer
    return result


def validate_generated_answer(
    generated: Dict[str, Any], evidence: List[EvidenceItem]
) -> tuple[str, List[Citation], List[str]]:
    generated = _normalize_generated_payload(generated)
    answer = str(generated.get("answer") or generated.get("text") or "").strip()
    warnings: List[str] = [str(item) for item in generated.get("warnings") or []]
    evidence_by_id = {item.evidence_id: item for item in evidence}
    aliases: Dict[str, List[str]] = {}
    def normalize_marker(value: Any) -> str:
        marker=str(value or "").strip().strip("[]").strip()
        if marker.casefold().startswith("evidence_id:"): marker=marker.split(":",1)[1].strip()
        return marker
    def add_alias(value: Any, evidence_id: str) -> None:
        alias = normalize_marker(value)
        if alias:
            aliases.setdefault(alias, [])
            if evidence_id not in aliases[alias]:
                aliases[alias].append(evidence_id)
    def add_nested_aliases(value: Any, evidence_id: str) -> None:
        if isinstance(value, dict):
            for key in ("evidence_id", "id", "chunk_id", "source_id"):
                add_alias(value.get(key), evidence_id)
            for child in value.values():
                add_nested_aliases(child, evidence_id)
        elif isinstance(value, list):
            for child in value:
                add_nested_aliases(child, evidence_id)
        elif isinstance(value, str):
            text = value.strip()
            for marker in re.findall(r'"id"\s*:\s*"([^"]+)"', text):
                add_alias(marker, evidence_id)
                suffix = re.search(r'_([0-9]+)$', marker)
                if suffix:
                    add_alias(suffix.group(1), evidence_id)
            if text.startswith("{") or text.startswith("["):
                try:
                    add_nested_aliases(json.loads(text), evidence_id)
                except (TypeError, ValueError):
                    pass

    for item in evidence:
        add_alias(item.evidence_id, item.evidence_id)
        add_alias(item.source_id, item.evidence_id)
        for key in ("source_evidence_ids", "evidence_ids"):
            for value in (item.metadata or {}).get(key) or []:
                add_alias(value, item.evidence_id)
        add_nested_aliases((item.metadata or {}).get("raw"), item.evidence_id)
    def resolve(value: Any) -> str | None:
        marker = normalize_marker(value)
        if marker in evidence_by_id:
            return marker
        matches = aliases.get(marker) or []
        if not matches and marker.isdigit() and 1 <= int(marker) <= len(evidence): return evidence[int(marker)-1].evidence_id
        return matches[0] if matches else None
    citation_values = list(generated.get("citations") or [])
    marker_ids = re.findall(r"\[([^\[\]]+)\]", answer)
    if not citation_values and answer:
        citation_values = marker_ids
    else:
        citation_values.extend(marker_ids)

    citations: List[Citation] = []
    seen: set[str] = set()
    for raw in citation_values:
        if isinstance(raw, str):
            evidence_id = raw.strip().strip("[]").strip()
            label = None
        else:
            evidence_id = str(raw.get("evidence_id") or "").strip().strip("[]").strip()
            label = raw.get("label")
        raw_evidence_id = evidence_id
        evidence_id = resolve(evidence_id)
        if evidence_id and evidence_id not in seen:
            source = evidence_by_id[evidence_id]
            citations.append(Citation(
                evidence_id=evidence_id,
                label=label or source.location_label or source.source_label or source.title,
            ))
            seen.add(evidence_id)
        elif evidence_id and evidence_id in seen:
            continue
        elif raw_evidence_id and raw_evidence_id != "?" and not re.fullmatch(r"[0-9a-fA-F]{16,}", raw_evidence_id):
            known_in_source = any(raw_evidence_id in str((item.metadata or {}).get("raw") or {}) for item in evidence)
            if not known_in_source:
                warnings.append(f"忽略未返回的证据引用：{raw_evidence_id}")

    def clean_marker(match: re.Match[str]) -> str:
        marker = normalize_marker(match.group(1))
        resolved = resolve(marker)
        if resolved:
            return f"[{resolved}]"
        if re.fullmatch(r"(?:[0-9a-fA-F]{16,}|[0-9]+)", marker):
            return ""
        return match.group(0)

    answer = re.sub(r"\[([^\[\]]+)\]", clean_marker, answer)
    if not answer:
        warnings.append("答案为空")
    if answer and citations:
        missing_markers = [
            f"[{citation.evidence_id}]"
            for citation in citations
            if f"[{citation.evidence_id}]" not in answer
        ]
        if missing_markers:
            answer = f"{answer.rstrip()} {' '.join(missing_markers)}"
    if answer and evidence and not citations:
        warnings.append("答案未提供有效证据引用")
    if answer and not evidence:
        warnings.append("答案缺少领域证据")
    return answer, citations, warnings
