"""输入模态和知识意图的低延迟、可重复基线分析器。"""

from __future__ import annotations

import re
from typing import List

from .schemas import InputAnalysis, QARequest


_INTENT_RULES = (
    ("MULTI_HOP", ("从", "到", "经过", "沿途", "路线", "如何到达")),
    ("LIST", ("哪些", "有几个", "列出", "清单", "分别")),
    ("COMPARE", ("比较", "对比", "区别", "哪个更", "差异")),
    ("TEMPORAL", ("何时", "哪年", "什么时候", "历史", "建于", "年代")),
    ("SPATIAL", ("哪里", "在哪", "位于", "位置", "地址", "附近", "周边", "周围", "相邻", "怎么去")),
    ("HIERARCHY", ("上级", "下级", "隶属", "属于", "组成", "包含", "包括", "里面有什么")),
    ("EXPLANATION", ("为什么", "原因", "意义", "作用", "介绍", "讲讲", "是什么")),
    ("IMAGE", ("图片", "照片", "图像", "图示", "看图", "配图", "长什么样")),
)

_QUESTION_MARKERS = (
    "为什么", "什么时候", "哪一年", "哪里", "在哪", "有哪些", "有几个",
    "怎么走", "如何到达", "怎么去", "是什么", "介绍", "位置", "地址",
)

_CHAT_EXACT = {
    "你好", "您好", "嗨", "hello", "hi", "在吗", "谢谢", "感谢", "再见",
    "拜拜", "早上好", "晚上好", "晚安", "好的", "好呀", "行", "可以",
    "明白了", "知道了", "嗯", "嗯嗯", "哈哈", "哈哈哈", "你是谁", "你叫什么",
}
_CHAT_FEEDBACK = re.compile(r"(?:这个答案|这个回答|你说的|你给的).*(?:什么东西|怎么回事|不对|不准确|离谱|看不懂)")
_CHAT_FULL = re.compile(
    r"(?:小v)?(?:你好|您好|嗨|hello|hi|在吗|你好吗|最近怎么样|早上好|晚上好|晚安|"
    r"谢谢|感谢|再见|拜拜|你是谁|你叫什么|陪我聊聊|和我聊聊|讲个笑话)"
)
_TRAILING_PARTICLES = re.compile(r"(?:在|是|有|的|位于|属于|关于)$")
_ENTITY_IMAGE_SUFFIX = re.compile(r"(?:的)?(?:图片|照片|图像|配图|外观)$")
_NON_ENTITY_PREFIXES = re.compile(r"^(?:请问|请介绍一下|介绍一下|告诉我|你不知道|你知道|我想知道|那|那么|然后|接着|另外|再|还有)")
_PRONOUNS = {"他", "她", "它", "他们", "她们", "这个", "那个", "这里", "那里", "该校", "该楼", "此处"}
_CURRENT_LOCATION_RE = re.compile(
    r"^(?:我(?:现在|当前)?|我的当前位置|当前位置|我所在的位置|这里|此处)"
    r"(?:在)?(?:哪|哪里|什么地方|什么位置)(?:吗)?$"
)


def is_current_location_question(message: str) -> bool:
    text = re.sub(r"[，。！？?；;、\s]+", "", str(message or "").strip())
    text = re.sub(r"^(?:请问|你知道|你能告诉我)", "", text)
    return bool(_CURRENT_LOCATION_RE.fullmatch(text))


def _usable_mention(value: str) -> bool:
    value = _NON_ENTITY_PREFIXES.sub("", value).strip()
    value = _ENTITY_IMAGE_SUFFIX.sub("", value).strip()
    return (
        bool(value)
        and value not in _PRONOUNS
        and not value.startswith(tuple(_PRONOUNS))
        and not all(char in _PRONOUNS for char in value)
        and not any(marker in value for marker in _QUESTION_MARKERS)
    )


def is_casual_message(message: str) -> bool:
    text = re.sub(r"[，。！？?；;、]+", "", str(message or "").strip()).casefold()
    if not text:
        return False
    return text in _CHAT_EXACT or bool(_CHAT_FULL.fullmatch(text)) or bool(_CHAT_FEEDBACK.search(text))


def _extract_mentions(message: str) -> List[str]:
    if is_current_location_question(message):
        return []
    text = re.sub(r"[，。！？?；;、]+", " ", str(message or "")).strip()
    mentions: List[str] = []
    for quoted in re.findall(r"[“\"]([^”\"]+)[”\"]", text):
        value = quoted.strip()
        if len(value) >= 2 and value not in mentions:
            mentions.append(value)
    for marker in _QUESTION_MARKERS:
        if marker in text:
            prefix = text.split(marker, 1)[0].strip()
            prefix = _NON_ENTITY_PREFIXES.sub("", prefix).strip()
            prefix = _TRAILING_PARTICLES.sub("", prefix).strip()
            prefix = _ENTITY_IMAGE_SUFFIX.sub("", prefix).strip()
            if 2 <= len(prefix) <= 32 and _usable_mention(prefix) and prefix not in mentions:
                mentions.append(prefix)
            break
    if not mentions:
        candidate = _NON_ENTITY_PREFIXES.sub("", text).strip()
        candidate = _TRAILING_PARTICLES.sub("", candidate).strip()
        candidate = _ENTITY_IMAGE_SUFFIX.sub("", candidate).strip()
        if 2 <= len(candidate) <= 32 and _usable_mention(candidate):
            mentions.append(candidate)
    return mentions[:5]


def _context_entity_names(conversation_context: dict | None) -> list[str]:
    context = conversation_context or {}
    names = []
    profile = context.get("long_term_memory") or {}
    if profile.get("last_entity"):
        names.append(str(profile["last_entity"]).strip())
    for turn in reversed(context.get("turns") or []):
        for value in turn.get("entities") or []:
            value = str(value or "").strip()
            if value and value not in names:
                names.append(value)
    return names[:3]


def analyze_input(request: QARequest, conversation_context: dict | None = None) -> InputAnalysis:
    message = str(request.message or "").strip()
    has_text = bool(message)
    has_image = bool(request.image_path or request.image_ids)
    if not has_text and not has_image:
        raise ValueError("message or image is required")

    if has_text and has_image:
        query_modality = "MULTIMODAL"
        modality_intent = "MULTIMODAL_TO_TEXT"
    elif has_image:
        query_modality = "IMAGE"
        modality_intent = "IMAGE_TO_IMAGE" if request.target_modality == "IMAGE" else "IMAGE_TO_TEXT"
    else:
        query_modality = "TEXT"
        modality_intent = "TEXT_TO_IMAGE" if request.target_modality == "IMAGE" else "TEXT_TO_TEXT"

    standalone_question = None
    if is_casual_message(message):
        intents = ["CHAT"]
        mentions = []
    else:
        intents = [name for name, markers in _INTENT_RULES if any(marker in message for marker in markers)]
        if not intents:
            intents = ["FACT"]
        mentions = _extract_mentions(message)
        context_names = _context_entity_names(conversation_context)
        followup = bool(re.search(r"^(?:那|那么|然后|接着|另外|再|还有)", message)) or any(
            token in message for token in _PRONOUNS
        )
        standalone_question = None
        if followup and context_names:
            if not mentions or any(token in message for token in _PRONOUNS):
                mentions = list(dict.fromkeys([*mentions, context_names[0]]))[:5]
            standalone_question = f"关于{context_names[0]}，{message}"
    return InputAnalysis(
        query_modality=query_modality,
        target_modality=request.target_modality or "TEXT",
        modality_intent=modality_intent,
        knowledge_intents=intents,
        mentions=mentions,
        has_text=has_text,
        has_image=has_image,
        standalone_question=standalone_question,
        intent_confidence=0.92 if intents == ["CHAT"] else 0.78,
        intent_source="rules+conversation" if standalone_question else "rules",
    )
