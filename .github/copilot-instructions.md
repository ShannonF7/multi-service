# Copilot Instructions for Zhangbi_Traveler (pgvector Project)

This specific sub-project (`DataBase/Search_Update_Context/json/pgvector/`) is the core backend for the "Zhangbi Traveler" AI assistant, providing RAG, LLM orchestration, and evaluation services.

## 🏗 Architecture & Core Modules
- **API Framework**: **FastAPI** (`app.py`). Use `BaseResponse` and `ErrorCode` for all standard API responses.
- **Database Layer (`src/database/`)**:
  - `vector.py`: Handles pgvector operations.
  - `models.py`: SQLAlchemy database models.
  - `session.py`: Database session management via `get_db` dependency.
- **Service Modules (`src/`)**:
  - `rag/`: Retrieval logic combining vector search and ranking.
  - `llm/`: Qwen/DashScope integration and streaming (`qwen_chat_stream`).
  - `intent/`: User intent recognition and routing.
  - `cv/`: Computer vision / image feature extraction.
- **Evaluation (`src/llm/dialogue_eval.py`)**: A state-machine based evaluation framework for testing dialogue flows.

## 💻 Coding Conventions

### 1. Configuration & Paths
- **Managed Settings**: Use `src/core/config.py` (inherits from `BaseSettings`). Add new parameters there rather than hardcoding.
- **Env**: Local configuration is stored in `.env` in the project root.

### 2. LLM & Prompting
- **API Access**: Use `call_api_with_retry` and `qwen_chat` (or `qwen_chat_stream`) from `src.llm.utils`.
- **Prompts**: Prompts are often templated in JSON (e.g., `scripts_template.json`).

### 3. Database & Models
- **SQLAlchemy**: Use async/sync sessions as defined in `session.py`.
- **Naming**: Follow existing Snake_Case for database columns and CamelCase for models.

### 4. API Patterns
- **Standard Response**: All routers should return `BaseResponse` with `ErrorCode`.
  ```python
  from src.scripts.schemas import BaseResponse, ErrorCode
  return BaseResponse(code=ErrorCode.SUCCESS.value, message="ok", data=result)
  ```
- **Error Handling**: Use `ErrorCode` (e.g., `PARAM_ERROR: 40001`, `INTERNAL_ERROR: 50000`). Raise `HTTPException` which is caught by global handlers in `app.py`.
- **Database Session**: Use sync sessions via `get_db` dependency (`SessionLocal` from `src.database.session`).

## 🛠 Workflows
- **Development**: Start with `uvicorn app:app --reload`.
- **Migrations**: Use `alembic` for database schema changes.
- **Testing**: Run tests using `pytest` (configs in `pytest.ini` or `.pytest_cache`). Use `run_min_tests.py` for a quick health check.
- **Logging**: Logs are stored in `logs/app.log`. Monitor this for runtime errors.

## 🧪 Evaluation Framework
- When working on dialogue logic, refer to `TaskState` (IDLE, IN_PROGRESS, COMPLETED, etc.) in `dialogue_eval.py` to ensure consistency with the tracking state machine.
