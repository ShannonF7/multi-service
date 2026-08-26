# global config（loading in env）
from pydantic_settings import BaseSettings
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

class Settings(BaseSettings):

    # LLM
    LLM_API_KEY: str
    LLM_MODEL: str = "qwen-plus"

    # RAG / Embedding
    EMBEDDING_MODEL_PATH: str
    retriever_top_k: int = 5

    # Security
    api_key: str
    allowed_ips: str = ""

    # 服务运行
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int

    # SSH Tunnel (Optional)
    USE_SSH_TUNNEL: bool = False
    SSH_HOST: str = ""
    SSH_PORT: int = 22
    SSH_USER: str = ""
    SSH_PASSWORD: str = ""
    REMOTE_DB_HOST: str = "127.0.0.1"
    REMOTE_DB_PORT: int = 5432

    class Config:
        env_file = os.path.join(BASE_DIR, ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
print("DB_HOST from settings:", settings.DB_HOST)