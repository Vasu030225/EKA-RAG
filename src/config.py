"""Load and validate application settings from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Configuration used by the document and question-answering pipeline."""

    data_dir: Path
    index_dir: Path
    embedding_model: str
    hf_token: str | None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    ollama_model: str
    ollama_base_url: str
    temperature: float


def _project_path(value: str) -> Path:
    """Resolve relative paths from the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> Settings:
    """Read settings from .env, applying defaults and basic validation."""
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        top_k = int(os.getenv("TOP_K", "4"))
        temperature = float(os.getenv("TEMPERATURE", "0"))
    except ValueError as exc:
        raise ValueError("CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, and TEMPERATURE must be numeric.") from exc

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be at least zero and smaller than CHUNK_SIZE.")
    if top_k <= 0:
        raise ValueError("TOP_K must be greater than zero.")
    if temperature < 0 or temperature > 2:
        raise ValueError("TEMPERATURE must be between 0 and 2.")

    return Settings(
        data_dir=_project_path(os.getenv("DATA_DIR", "./data")),
        index_dir=_project_path(os.getenv("INDEX_DIR", "./faiss_index")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        hf_token=os.getenv("HF_TOKEN") or None,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )
