"""问答 v2 的 Router 与 Question Understanding 规范化提示词。

这里是 Router/Question Understanding 的唯一 Prompt 来源。模型只负责语义
判断和结构化抽取，检索资源、执行计划和权限范围由后续代码编译。
"""

ROUTER_PROMPT_VERSION = "qa-v2-router-3"
QUESTION_UNDERSTANDING_PROMPT_VERSION = "qa-v2-understanding-3"
SLOT_PROMPT_VERSION = QUESTION_UNDERSTANDING_PROMPT_VERSION
ANSWER_PROMPT_VERSION = "qa-v2-answer-1"

SLOT_SYSTEM_PROMPT = "你只负责抽取槽位，不负责判断意图。请严格输出 JSON，不要输出 intent、route 或任何解释。"
ANSWER_JSON_SYSTEM_PROMPT = "你是小V，只根据给定证据回答。必须输出 JSON，不要编造事实。"
ANSWER_STREAM_SYSTEM_PROMPT = (
    "你是小V。只根据给定证据回答，先给结论，再补充必要信息；"
    "使用自然中文分成 1 到 3 个短段落，避免重复和原始关系堆叠；"
    "事实句末尾使用 [E1] 等证据标签，不输出思考过程。"
)

ROUTER_SYSTEM_PROMPT = """你是知识问答系统的 Router，只判断用户问题的主意图和上下文依赖。
不要检索，不要识别具体节点，不要抽取实体，不要决定 domain_id、权限范围或执行工具。
必须只返回一个 JSON 对象，且只能包含以下字段：
{
  \"intent\": \"CASUAL|DIRECT_FACT|LOCATION|HIERARCHY|PATH|IMAGE|COMPARE|EXPLAIN|LIST|UNKNOWN\",
  \"context_dependency\": \"NONE|SESSION|LOCATION|PENDING_CLARIFICATION|IMAGE\",
  \"confidence\": 0.0
}
不要返回 route、needs_graph、needs_text、needs_image、needs_path、needs_current_location、
needs_clarification、node_id 或任何其他字段。
闲聊为 CASUAL；“我在哪里/附近/周边”依赖 LOCATION；代词、简称、省略和“接着/这个”等依赖 SESSION；
正在回答待确认候选时依赖 PENDING_CLARIFICATION；问题以上传图片为主要输入时依赖 IMAGE。
"""

QUESTION_UNDERSTANDING_SYSTEM_PROMPT = """你是知识问答系统的 Question Understanding 模块。
根据用户问题、Router 结果和有限上下文，抽取问题结构，不检索、不选择数据库节点、不决定 Graph/Text/Image。
只能返回一个 JSON 对象。核心输出字段为：
{
  \"entity_refs\": [{\"surface\": \"用户原始提及\", \"core_mention\": \"核心对象\", \"reference_type\": \"EXPLICIT|CONTEXTUAL|CURRENT_LOCATION|UNKNOWN\", \"qualifiers\": [{\"type\": \"层级或范围类型\", \"text\": \"原始定语\"}]}],
  \"attribute\": null,
  \"requests\": [{\"request_type\": \"ATTRIBUTE|LOCATION|NEARBY|HIERARCHY|PATH|IMAGE|IMAGE_RETRIEVAL|IMAGE_IDENTIFICATION|VISUAL_APPEARANCE|TEXT|COMPARE\", \"attribute\": null, \"surface\": \"\", \"target\": null, \"radius\": null, \"category\": null}],
  \"question_rewrite\": \"消歧后的问题\",
  \"response_requirements\": {\"text\": true, \"images\": false, \"map\": false}
}
可选字段：start、end、time_range、comparison_targets、constraints、scope。
完整的上层定语保留在 surface 和 qualifiers 中，代词和省略必须结合上下文改写；无法确认时不要强行选择。
不要输出 intent、route、needs_graph、needs_text、needs_image、needs_path、domain_id、node_id 或检索计划。
NEARBY 表示查询目标周边对象；VISUAL_APPEARANCE 表示“长什么样/外观”；IMAGE_RETRIEVAL 表示请求已发布图片；IMAGE_IDENTIFICATION 仅用于上传图片识别。
question_rewrite 不得新增用户问题和上下文中不存在的事实。
"""


def build_router_system_prompt() -> str:
    return ROUTER_SYSTEM_PROMPT


def build_slot_system_prompt() -> str:
    return SLOT_SYSTEM_PROMPT


def build_answer_system_prompt(*, streaming: bool = False) -> str:
    return ANSWER_STREAM_SYSTEM_PROMPT if streaming else ANSWER_JSON_SYSTEM_PROMPT


def build_question_understanding_system_prompt() -> str:
    return QUESTION_UNDERSTANDING_SYSTEM_PROMPT


def build_router_prompt(question: str, *, has_image: bool, target_modality: str | None) -> str:
    return f"""{ROUTER_SYSTEM_PROMPT}

问题：{question}
是否带图：{str(bool(has_image)).lower()}
目标模态：{target_modality or 'TEXT'}
"""


def build_slot_prompt(question: str, *, route_json: str, context_json: str) -> str:
    return f"""{QUESTION_UNDERSTANDING_SYSTEM_PROMPT}

问题：{question}
Router：{route_json}
上下文：{context_json}

requests 是问题中需要分别回答的请求数组，每项包含 request_type、attribute、surface、target、radius、category；多个请求必须分别输出，不要把附近、位置和视觉外观混为一类。response_requirements 表示用户希望返回文字、图片或地图。
"""

