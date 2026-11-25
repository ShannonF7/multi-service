"""QA v2 的答案生成和引用校验。

答案模型只消费 QuerySpec 与 Evidence，不访问数据库、不重新选择领域或检索计划。
引用先做确定性结构校验，再可选调用 DecisionModel 做事实句支持性判断。
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib import request as urllib_request

from pydantic import BaseModel, Field

from src.kg_qa_shared.budget import BudgetSection, ContextBudgetManager
from ..models import QuerySpec
from .decision_model import DecisionModel
from .prompts import build_answer_system_prompt
from .request_catalog import attribute_aliases, attribute_display
from .retrieval import Evidence


class CitationRecord(BaseModel):
    label: str
    evidence_id: str
    status: str = "STRUCTURAL"
    confidence: float = 1.0
    reason: str = ""


class AnswerDraft(BaseModel):
    answer: str = ""
    citation_labels: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    runtime_meta: dict[str, Any] = Field(default_factory=dict)


class ValidatedAnswer(BaseModel):
    answer: str
    citations: list[CitationRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    runtime_meta: dict[str, Any] = Field(default_factory=dict)


class AnswerGenerator:
    """使用注入生成函数或 OpenAI-compatible 本地 Qwen 生成答案。"""

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        generate_fn: Optional[Callable[..., Any]] = None,
        request_fn: Optional[Callable[..., Any]] = None,
        context_budget: Optional[ContextBudgetManager] = None,
    ) -> None:
        configured_provider = provider or os.getenv("ANSWER_PROVIDER") or "local"
        self.provider = _normalize_answer_provider(configured_provider)
        self._provider_explicit = provider is not None
        if self.provider == "openai_compatible":
            configured_base_url = (
                base_url
                if self._provider_explicit
                else (
                    os.getenv("ANSWER_API_BASE_URL")
                    or os.getenv("ANSWER_BASE_URL")
                    or os.getenv("OPENAI_BASE_URL")
                    or base_url
                )
            )
            configured_model = (
                model
                if self._provider_explicit
                else (
                    os.getenv("ANSWER_API_MODEL")
                    or os.getenv("ANSWER_MODEL")
                    or os.getenv("LLM_MODEL")
                    or model
                )
            )
            self.api_key = api_key or os.getenv("ANSWER_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            configured_base_url = base_url or os.getenv("LOCAL_ANSWER_BASE_URL") or os.getenv("LOCAL_DECISION_BASE_URL")
            configured_model = model or os.getenv("LOCAL_ANSWER_MODEL") or os.getenv("LOCAL_DECISION_MODEL")
            self.api_key = api_key
        self.base_url = (configured_base_url or "").rstrip("/")
        self.model = configured_model or ""
        self.timeout = float(timeout)
        try:
            configured_output_tokens = int(os.getenv("ANSWER_MAX_OUTPUT_TOKENS", "256"))
        except ValueError:
            configured_output_tokens = 768
        self.max_output_tokens = max(64, min(configured_output_tokens, 4096))
        self.generate_fn = generate_fn
        self.request_fn = request_fn
        self.context_budget = context_budget or ContextBudgetManager.from_env()

    def generate(
        self,
        question: str,
        *,
        spec: QuerySpec,
        evidence: list[Evidence],
    ) -> AnswerDraft:
        spec_json = json.dumps(spec.model_dump(), ensure_ascii=False)
        question_spec_tokens = (
            self.context_budget.estimate_tokens(question)
            + self.context_budget.estimate_tokens(spec_json)
        )
        output_reserve = max(0, self.max_output_tokens - self.context_budget.reserved_output_tokens)
        # 先扣除问题和 QuerySpec，再为答案额外预留空间；至少保留 64 个输入 token
        # 给证据，避免生产环境的输出上限让小窗口完全丢失证据上下文。
        evidence_floor = 64
        output_reserve = min(
            output_reserve,
            max(0, self.context_budget.input_budget_tokens - question_spec_tokens - evidence_floor),
        )
        reserved_tokens = question_spec_tokens + output_reserve
        context = _evidence_context(
            evidence,
            budget_manager=self.context_budget,
            extra_reserved_tokens=reserved_tokens,
        )
        prompt = _build_answer_prompt(question, spec, context)
        generation_started = time.perf_counter()
        raw = self.generate_fn(prompt=prompt, response_format="json") if self.generate_fn else self._request(prompt)
        runtime_meta = _runtime_meta(
            raw,
            provider=self.provider,
            model=self.model,
            elapsed_ms=int((time.perf_counter() - generation_started) * 1000),
        )
        payload = _as_dict(raw)
        answer = str(payload.get("answer") or "").strip()
        labels = payload.get("citation_labels") or _labels_from_answer(answer)
        if isinstance(labels, str):
            labels = [labels]
        return AnswerDraft(
            answer=answer,
            citation_labels=[str(item).strip() for item in labels if str(item).strip()],
            runtime_meta=runtime_meta,
        )

    def _request(self, prompt: str) -> Any:
        if not self.base_url or not self.model:
            if self.provider == "openai_compatible":
                raise RuntimeError("答案 API 未配置 ANSWER_API_BASE_URL/ANSWER_API_MODEL")
            raise RuntimeError("答案模型未配置 LOCAL_ANSWER_BASE_URL/LOCAL_ANSWER_MODEL")
        if self.provider == "openai_compatible" and not self.api_key:
            raise RuntimeError("答案 API 未配置 ANSWER_API_KEY/LLM_API_KEY")
        body = json.dumps({
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": build_answer_system_prompt(),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.request_fn:
            return self.request_fn(
                endpoint=endpoint,
                data=body,
                headers=headers,
                timeout=self.timeout,
            )
        req = urllib_request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib_request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def _normalize_answer_provider(value: str) -> str:
    provider = str(value or "").strip().lower().replace("-", "_")
    if provider in {"local", "vllm"}:
        return "local"
    if provider in {"api", "openai", "openai_compatible"}:
        return "openai_compatible"
    raise ValueError("不支持的答案模型 provider；可选 local 或 openai_compatible")


def fast_casual_answer(question: str) -> Optional[AnswerDraft]:
    """为高确定性社交请求提供本地答案，避免闲聊再次调用 7B。"""
    text = re.sub(r"[\s，。！？?；;、]+", "", str(question or "")).casefold()
    if any(marker in text for marker in ("讲个笑话", "说个笑话", "讲笑话", "说笑话")):
        return AnswerDraft(answer="为什么电脑经常生病？因为它的 Windows 太多了！")
    if any(marker in text for marker in ("讲个故事", "说个故事")):
        return AnswerDraft(answer="从前有一只总爱迷路的小猫，后来它学会了记住沿途的风景，于是每一次回家都变成了一次小小的探险。")
    if text in {"你好", "您好", "嗨", "hello", "hi"}:
        return AnswerDraft(answer="你好！我是小V，很高兴和你聊天。")
    if text in {"谢谢", "感谢"}:
        return AnswerDraft(answer="不客气，随时可以找我。")
    if text in {"再见"}:
        return AnswerDraft(answer="再见，祝你今天一切顺利！")
    if re.match(r"^(?:我|今天|最近)?(?:好|有点|太|真的)?(?:累|困|疲惫|难过|开心|郁闷|烦|焦虑|紧张)(?:啊|呀|了|呢)?$", text):
        return AnswerDraft(answer="听起来你今天有点累。先休息一下吧，想聊聊什么都可以。")
    if any(marker in text for marker in ("心情怎么样", "心情如何", "心情好吗")):
        return AnswerDraft(answer="我没有真正的情绪，不过我一直在线，可以陪你聊聊。")
    return None


class CitationValidator:
    """校验证据 ID，并可选调用决策模型判断事实句是否被支持。"""

    def __init__(self, decision_model: Optional[DecisionModel] = None) -> None:
        self.decision_model = decision_model

    def validate(
        self,
        draft: AnswerDraft,
        evidence: list[Evidence],
        *,
        semantic: bool = True,
    ) -> ValidatedAnswer:
        if not evidence:
            return ValidatedAnswer(
                answer=_strip_citation_markers(draft.answer),
                citations=[],
                warnings=[],
                claims=draft.claims,
                runtime_meta=draft.runtime_meta,
            )
        by_label = {f"E{index}": item for index, item in enumerate(evidence, start=1)}
        answer = _remove_unknown_markers(draft.answer, set(by_label))
        labels = _labels_from_answer(answer) + [
            label for label in draft.citation_labels if label in by_label
        ]
        labels = list(dict.fromkeys(labels))
        warnings: list[str] = []
        if draft.citation_labels and len(labels) < len(set(draft.citation_labels)):
            warnings.append("答案中存在无法对应当前证据的引用，已忽略")
        if evidence and not labels and answer:
            labels = ["E1"]
            answer = f"{answer} [{labels[0]}]"
            warnings.append("答案未给出引用标记，已补充最相关证据")
        citations: list[CitationRecord] = []
        for label in labels:
            item = by_label[label]
            citations.append(CitationRecord(label=label, evidence_id=item.evidence_id))

        if semantic and self.decision_model is not None and citations:
            citations = self._semantic_validate(answer, citations, by_label)
        return ValidatedAnswer(
            answer=answer,
            citations=citations,
            warnings=warnings,
            claims=draft.claims,
            runtime_meta=draft.runtime_meta,
        )

    def _semantic_validate(
        self,
        answer: str,
        citations: list[CitationRecord],
        by_label: Mapping[str, Evidence],
    ) -> list[CitationRecord]:
        result: list[CitationRecord] = []
        for sentence in _sentences(answer):
            labels = _labels_from_answer(sentence)
            if not labels:
                continue
            for label in labels:
                evidence = by_label[label]
                try:
                    raw = self.decision_model.decide(
                        "citation_validation",
                        {
                            "answer_sentence": sentence,
                            "evidence": [{
                                "evidence_id": evidence.evidence_id,
                                "content": evidence.content,
                            }],
                        },
                    )
                    status = str(raw.get("status") or "STRUCTURAL").upper()
                    if status not in {"ENTAILED", "PARTIAL", "NOT_ENTAILED"}:
                        status = "STRUCTURAL"
                    confidence = _clamp(raw.get("confidence"))
                    if status == "NOT_ENTAILED":
                        continue
                    result.append(CitationRecord(
                        label=label,
                        evidence_id=evidence.evidence_id,
                        status=status,
                        confidence=confidence,
                        reason=str(raw.get("reason") or ""),
                    ))
                except Exception:
                    result.append(CitationRecord(
                        label=label,
                        evidence_id=evidence.evidence_id,
                    ))
        return result or citations



def fast_nearby_answer(spec: QuerySpec, evidence: list[Evidence]) -> Optional[AnswerDraft]:
    """用空间查询证据直接组织周边答案，避免不必要的大模型调用。"""
    if not any(item.request_type == "NEARBY" for item in spec.requests):
        return None
    nearby_items = []
    labels = []
    for index, item in enumerate(evidence, start=1):
        if item.source_type != "graph":
            continue
        distance_m = (item.metadata or {}).get("distance_m")
        if distance_m in (None, ""):
            continue
        nearby_items.append(str(item.content or "").strip())
        labels.append(f"E{index}")
    if nearby_items:
        answer = "附近可以看到：\n\n" + "\n".join(
            f"- {content} [{label}]" for content, label in zip(nearby_items, labels)
        )
        return AnswerDraft(answer=answer, citation_labels=labels)
    return AnswerDraft(answer="在当前已发布图谱和默认周边范围内，暂未找到可展示的附近位置。")


def fast_graph_answer(spec: QuerySpec, evidence: list[Evidence]) -> Optional[AnswerDraft]:
    """为已对齐的简单位置问题直接组织可引用答案，不调用生成模型。"""
    if spec.needs_current_location:
        for index, item in enumerate(evidence, start=1):
            if item.source_type != "graph":
                continue
            content = str(item.content or "").strip()
            if content:
                root = (item.metadata or {}).get("root") or {}
                name = str(root.get("name") or "").strip()
                if name and "不在" not in content:
                    content = f"你当前定位在“{name}”。"
                return AnswerDraft(answer=f"{content} [E{index}]", citation_labels=[f"E{index}"])
        return AnswerDraft(answer="当前没有可核验的领域内定位信息。")
    if spec.intent not in {"LOCATION", "HIERARCHY"}:
        return None
    if not spec.constraints.get("resolved_node_ids"):
        return None

    hierarchy_item: tuple[int, Evidence] | None = None
    coordinate_item: tuple[int, Evidence] | None = None
    fallback_item: tuple[int, Evidence] | None = None
    for index, item in enumerate(evidence, start=1):
        if item.source_type != "graph":
            continue
        content = str(item.content or "").strip()
        if not content:
            continue
        if fallback_item is None and any(token in content for token in ("位于", "属于", "上级", "下级")):
            fallback_item = (index, item)
        if hierarchy_item is None and "层级位置" in content:
            hierarchy_item = (index, item)
        if coordinate_item is None and "坐标" in content:
            coordinate_item = (index, item)

    if hierarchy_item is not None:
        index, item = hierarchy_item
        metadata = item.metadata or {}
        path = [
            str(value).strip()
            for value in (metadata.get("hierarchy_path") or [])
            if str(value).strip()
        ]
        root = metadata.get("root") if isinstance(metadata.get("root"), dict) else {}
        name = str(root.get("name") or "").strip()
        if not name and spec.constraints.get("resolved_candidates"):
            candidate = spec.constraints["resolved_candidates"][0]
            if isinstance(candidate, dict):
                name = str(candidate.get("name") or "").strip()
        if path and name and path[-1] == name:
            path = path[:-1]
        if name and path:
            answer = f"{name}位于{' · '.join(path)}。"
        else:
            answer = re.sub(r"的层级位置为：", "位于", str(item.content), count=1)
            answer = answer.rstrip("。") + "。"
        labels = [f"E{index}"]
        parts = [f"{answer} [E{index}]"]
        if coordinate_item is not None:
            coordinate_index, coordinate = coordinate_item
            if coordinate_index != index:
                parts.append(f"{coordinate.content.rstrip('。')}。 [E{coordinate_index}]")
                labels.append(f"E{coordinate_index}")
        return AnswerDraft(answer="\n\n".join(parts), citation_labels=labels)

    if fallback_item is not None:
        index, item = fallback_item
        return AnswerDraft(
            answer=f"{item.content.rstrip('。')}。 [E{index}]",
            citation_labels=[f"E{index}"],
        )
    return None



def fast_visual_answer(spec: QuerySpec, evidence: list[Evidence]) -> Optional[AnswerDraft]:
    """节点已对齐时直接返回层级与已发布图片，不调用答案模型。"""
    request_types = {item.request_type for item in spec.requests}
    if not request_types & {"VISUAL_APPEARANCE", "IMAGE_RETRIEVAL"}:
        return None
    graph_parts: list[str] = []
    image_labels: list[str] = []
    labels: list[str] = []
    display_name = str(spec.targets[0] if spec.targets else "该位置").strip() or "该位置"
    location_requested = "LOCATION" in request_types
    first_graph_label: str | None = None
    for index, item in enumerate(evidence, start=1):
        if item.source_type == "graph":
            first_graph_label = first_graph_label or f"E{index}"
            content = str(item.content or "").strip()
            if content and ("层级位置" in content or "位于" in content or "属于" in content or "Located In" in content or "层级归属" in content or "Contains" in content):
                if " Located In " in content:
                    left, right = content.split(" Located In ", 1)
                    content = f"{left.strip()}位于{right.strip()}"
                elif " 层级归属 " in content:
                    parent, child = content.split(" 层级归属 ", 1)
                    content = f"{child.strip()}位于{parent.strip()}"
                elif " Contains " in content:
                    parent, child = content.split(" Contains ", 1)
                    content = f"{child.strip()}位于{parent.strip()}"
                graph_parts.append(content.rstrip("。") + f"。 [E{index}]")
                labels.append(f"E{index}")
                root = (item.metadata or {}).get("root") or {}
                display_name = str(root.get("name") or display_name).strip()
        elif item.source_type == "image":
            image_labels.append(f"E{index}")
            labels.append(f"E{index}")
    if location_requested and not graph_parts:
        for candidate in spec.constraints.get("resolved_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            name = str(candidate.get("name") or display_name).strip()
            path = [str(value).strip() for value in (candidate.get("hierarchy_path") or []) if str(value).strip()]
            if path and path[-1] == name:
                path = path[:-1]
            if path:
                display_name = name
                graph_parts.append(f"{name}位于{' · '.join(path)}。 [{first_graph_label or 'E1'}]")
                labels.append(first_graph_label or "E1")
                break
    paragraphs = graph_parts[:1]
    if image_labels:
        paragraphs.append(f"我找到了{display_name}的已发布图片，先给你看这几张。")
    elif not paragraphs:
        return None
    return AnswerDraft(answer="\n\n".join(paragraphs), citation_labels=list(dict.fromkeys(labels)))


def fast_fact_answer(spec: QuerySpec, evidence: list[Evidence]) -> Optional[AnswerDraft]:
    """Bind each requested attribute to its own graph evidence."""
    if spec.intent != "DIRECT_FACT" or not spec.requests:
        return None
    terms = {key: attribute_aliases(key) for key in (
        "manager", "contact", "phone", "email", "area", "name"
    )}
    display = {key: attribute_display(key) for key in terms}
    requests = [item for item in spec.requests if item.request_type == "ATTRIBUTE" and item.attribute]
    if not requests:
        return None
    paragraphs, labels, claims = [], [], []
    for request in requests:
        attribute = str(request.attribute)
        matched, matched_index = None, 0
        for index, item in enumerate(evidence, start=1):
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            fact = metadata.get("fact") if isinstance(metadata.get("fact"), dict) else {}
            relation = metadata.get("relation") if isinstance(metadata.get("relation"), dict) else {}
            key = str(fact.get("key") or fact.get("formal_property_id") or relation.get("relation_label") or relation.get("relation_code") or "")
            value = str(fact.get("value") or relation.get("target_name") or "").strip()
            if value and any(term.casefold() in key.casefold() or term.casefold() in str(item.content).casefold() for term in terms.get(attribute, (attribute,))):
                matched, matched_index = value, index
                break
        if matched:
            text = f"{display.get(attribute, attribute)}是{matched}。 [E{matched_index}]"
            paragraphs.append(text)
            labels.append(f"E{matched_index}")
            claims.append({"request": attribute, "status": "FOUND", "text": text, "value": matched, "evidence_ids": [evidence[matched_index - 1].evidence_id]})
        else:
            text = f"当前已发布信息中暂未找到{display.get(attribute, attribute)}信息。"
            paragraphs.append(text)
            claims.append({"request": attribute, "status": "INSUFFICIENT_EVIDENCE", "text": text, "value": None, "evidence_ids": []})
    return AnswerDraft(answer="\n\n".join(paragraphs), citation_labels=list(dict.fromkeys(labels)), claims=claims)


def fast_attribute_answer(
    spec: QuerySpec,
    evidence: list[Evidence],
) -> Optional[AnswerDraft]:
    """从已发布节点事实或关系中直接回答简单属性问题。"""
    if spec.requests:
        return fast_fact_answer(spec, evidence)
    if spec.intent != "DIRECT_FACT":
        return None
    labels = {
        key: (attribute_display(key), attribute_aliases(key))
        for key in ("manager", "contact", "phone", "email", "name")
    }
    config = labels.get(spec.attribute)
    if config is None:
        return None
    default_label, terms = config
    for index, item in enumerate(evidence, start=1):
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        fact = metadata.get("fact") if isinstance(metadata.get("fact"), dict) else {}
        relation = metadata.get("relation") if isinstance(metadata.get("relation"), dict) else {}
        key = str(fact.get("key") or relation.get("relation_label") or "").strip()
        value = str(fact.get("value") or relation.get("target_name") or "").strip()
        content = str(item.content or "").strip()
        if not value:
            continue
        if not any(term in key or term in content for term in terms):
            continue
        label = default_label
        if spec.attribute == "manager" and "负责人" in f"{key}{content}":
            label = "负责人"
        return AnswerDraft(
            answer=f"该位置的{label}是{value}。 [E{index}]",
            citation_labels=[f"E{index}"],
        )
    return None


def fast_attribute_unavailable_answer(
    spec: QuerySpec,
    evidence: list[Evidence],
) -> Optional[AnswerDraft]:
    """属性证据不存在时直接给出受限答案，避免模型拿无关证据填充。"""
    labels = {
        key: (attribute_display(key), attribute_aliases(key))
        for key in ("manager", "contact", "phone", "email", "name")
    }
    if spec.intent != "DIRECT_FACT" or spec.attribute not in labels:
        return None
    label, terms = labels[spec.attribute]
    if any(
        any(term in str(item.content or "") for term in terms)
        for item in evidence
    ):
        return None
    return AnswerDraft(
        answer=f"当前领域的已发布图谱中，暂未记录该位置的{label}信息。",
        citation_labels=[],
    )


def _build_answer_prompt(question: str, spec: QuerySpec, evidence_context: str) -> str:
    separator = chr(10)
    if spec.route == "CASUAL":
        return (
            "这是闲聊，请自然回答，不需要检索、引用或证据标签。"
            "不要输出 [E1]、[E2] 等引用标记，只输出 JSON。"
            + separator
            + f"用户问题：{question}"
            + separator
            + '只输出 JSON：{"answer":"...","citation_labels":[]}'
        )
    image_rule = ""
    if spec.intent == "IMAGE" and spec.constraints.get("image_path"):
        image_rule = (
            "这是上传图片识别问题。若证据中有图片匹配节点，必须先写“根据上传图片，"
            "最匹配的已发布图谱节点是……”；不要把图片识别结果写成“当前位置”，"
            "除非用户明确询问 GPS 当前所在位置。"
        )
    return (
        "只能使用下面的证据回答。先给直接结论，再按需要补充层级、属性、关系或图片信息。"
        "答案使用自然中文，分成 1 到 3 个短段落；不要把原始关系逐条堆叠，不要重复同一事实。"
        "位置问题优先写成“对象位于……”，不要出现“实体”“检索过程”等技术词。"
        + image_rule
        + "每个事实句末尾使用证据标签 [E1]、[E2]，标签只能来自给定证据。不要编造事实。"
        + separator
        + f"用户问题：{question}"
        + separator
        + f"问题规格：{json.dumps(spec.model_dump(), ensure_ascii=False)}"
        + separator
        + "证据："
        + separator
        + evidence_context
        + separator
        + '只输出 JSON：{"answer":"...","citation_labels":["E1"]}'
    )


def _strip_citation_markers(answer: str) -> str:
    value = str(answer or "")
    value = re.sub(r"\s*\[(?:evidence_id:\s*)?(?:graph|text|image):[^\]]+\]", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*\[E\d+\]", "", value, flags=re.IGNORECASE)
    return value.strip()


def _evidence_context(
    evidence: list[Evidence],
    *,
    budget_manager: Optional[ContextBudgetManager] = None,
    extra_reserved_tokens: int = 0,
) -> str:
    if not evidence:
        return "暂无可用证据。"
    if budget_manager is None:
        chunks = []
        for index, item in enumerate(evidence, start=1):
            chunks.append(
                f"[E{index}] evidence_id={item.evidence_id}; source_type={item.source_type}; "
                f"content={item.content}; source_url={item.source_url or ''}"
            )
        return "\n".join(chunks)

    section_limit = max(
        96,
        budget_manager.input_budget_tokens // min(max(len(evidence), 1), 8),
    )
    sections = []
    for index, item in enumerate(evidence, start=1):
        sections.append(BudgetSection(
            name=f"evidence:{index}",
            text=(
                f"[E{index}] evidence_id={item.evidence_id}; source_type={item.source_type}; "
                f"content={item.content}; source_url={item.source_url or ''}"
            ),
            priority={"graph": 100, "text": 80, "image": 70}.get(item.source_type, 60),
            max_tokens=section_limit,
        ))
    result = budget_manager.pack(
        sections,
        extra_reserved_tokens=max(0, int(extra_reserved_tokens)),
    )
    return result.text or "暂无可用证据。"


def _labels_from_answer(answer: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[(E\d+)\]", str(answer or ""))))


def _remove_unknown_markers(answer: str, allowed: set[str]) -> str:
    value = re.sub(
        r"\s*\[(?:evidence_id:\s*)?(?:graph|text|image):[^\]]+\]",
        "",
        str(answer or ""),
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1).upper() in allowed else ""

    return re.sub(r"\s*\[(E\d+)\]", replace, value, flags=re.IGNORECASE).strip()


def _sentences(answer: str) -> Iterable[str]:
    parts: list[str] = []
    for raw in re.split(r"(?<=[。！？!?])\s*|\n+", answer):
        item = raw.strip()
        if not item:
            continue
        leading = re.match(r"^(?P<labels>(?:\[E\d+\]\s*)+)(?P<rest>.*)$", item)
        if leading and parts:
            parts[-1] = f"{parts[-1]} {leading.group('labels').strip()}".strip()
            item = leading.group("rest").strip()
        if item:
            parts.append(item)
    return iter(parts)


def _runtime_meta(
    raw: Any,
    *,
    provider: str,
    model: str,
    elapsed_ms: int,
) -> dict[str, Any]:
    """提取供应商返回的 usage/finish_reason，供响应和审计使用。"""
    meta: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "total_ms": max(0, int(elapsed_ms)),
    }
    if not isinstance(raw, dict):
        return meta
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    choices = raw.get("choices") or []
    if choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
        if finish_reason:
            meta["finish_reason"] = str(finish_reason)
    aliases = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "total_tokens": "total_tokens",
    }
    for source, target in aliases.items():
        value = usage.get(source)
        if value is not None and target not in meta:
            meta[target] = value
    return meta


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        choices = value.get("choices")
        if choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if content is not None:
                return _as_dict(content)
        return value
    if isinstance(value, str):
        parsed = json.loads(value.strip())
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("答案模型必须返回 JSON 对象")


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
