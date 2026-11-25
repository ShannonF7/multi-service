"""V1 会话模块的兼容导入层，实际实现由 V1 与 V2 共用。"""

from src.kg_qa_shared.memory import build_summary_contract
from src.kg_qa_shared.session import (
    ConversationContext,
    ConversationSessionStore,
    PersistentConversationSessionStore,
    build_default_session_store,
)

__all__ = [
    "ConversationContext",
    "ConversationSessionStore",
    "PersistentConversationSessionStore",
    "build_default_session_store",
    "build_summary_contract",
]
