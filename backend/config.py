import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL        = os.getenv("FORGE_BASE_URL", "http://localhost:8080")
API_KEY         = os.getenv("FORGE_API_KEY",  "sk-no-key")
MODEL_NAME      = os.getenv("FORGE_MODEL_NAME", "local-model")
DB_PATH         = os.getenv("FORGE_DB_PATH",  "forge.db")

# Approval gate + subagent orchestration — kept deliberately separate
# from forge.db (see backend/checkpointer.py).
CHECKPOINTS_DB_PATH      = os.getenv("FORGE_CHECKPOINTS_DB_PATH", "checkpoints.db")
APPROVAL_TIMEOUT_SECONDS = int(os.getenv("FORGE_APPROVAL_TIMEOUT_SECONDS", "300"))

# Informational only in this codebase: the actual concurrency limit is
# set on llama-server itself (its own --parallel flag), not read here.
# Kept so the number lives in one documented place.
SUBAGENT_PARALLEL = int(os.getenv("FORGE_SUBAGENT_PARALLEL", "4"))
