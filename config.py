import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    chroma_path: str
    data_path: str
    database_url: str
    chat_model: str
    summary_model: str
    embedding_model: str
    openai_vision_model: str | None
    image_to_text_provider: str
    retrieval_top_k: int
    retrieval_score_threshold: float
    recent_history_messages: int
    summary_trigger_messages: int
    max_summary_tokens: int


def get_settings() -> Settings:
    provider = os.getenv("IMAGE_TO_TEXT_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "llava"}:
        raise RuntimeError("IMAGE_TO_TEXT_PROVIDER must be either 'openai' or 'llava'.")

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/multimodal_rag",
    )

    return Settings(
        chroma_path=os.getenv("CHROMA_PATH", "chroma"),
        data_path=os.getenv("DATA_PATH", "data/raw"),
        database_url=database_url,
        chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
        summary_model=os.getenv("SUMMARY_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_vision_model=os.getenv("OPENAI_VISION_MODEL"),
        image_to_text_provider=provider,
        retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 4),
        retrieval_score_threshold=_get_float("RETRIEVAL_SCORE_THRESHOLD", 0.7),
        recent_history_messages=_get_int("RECENT_HISTORY_MESSAGES", 6),
        summary_trigger_messages=_get_int("SUMMARY_TRIGGER_MESSAGES", 4),
        max_summary_tokens=_get_int("MAX_SUMMARY_TOKENS", 300),
    )
