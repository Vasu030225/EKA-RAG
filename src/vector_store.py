"""Create, persist, and reuse a local FAISS vector index."""

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.chunker import split_documents
from src.config import Settings
from src.loader import load_pdf_documents

logger = logging.getLogger(__name__)


def _create_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Initialize the configured local Hugging Face embedding model."""
    model_kwargs = {"token": settings.hf_token} if settings.hf_token else {}
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs=model_kwargs,
    )


def index_exists(index_dir: Path) -> bool:
    """Return whether both files that make up a saved FAISS index exist."""
    return (index_dir / "index.faiss").is_file() and (index_dir / "index.pkl").is_file()


def build_index(chunks: list[Document], settings: Settings) -> FAISS:
    """Build and save a FAISS index from document chunks."""
    if not chunks:
        raise ValueError("Cannot build a FAISS index because no text chunks were produced.")

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, _create_embeddings(settings))
    vector_store.save_local(str(settings.index_dir))
    logger.info("Built and saved FAISS index at %s.", settings.index_dir)
    return vector_store


def load_index(settings: Settings) -> FAISS:
    """Load the saved local FAISS index."""
    if not index_exists(settings.index_dir):
        raise FileNotFoundError(
            f"No complete FAISS index found in {settings.index_dir}. "
            "Run with --rebuild to create one."
        )

    # This pickle belongs to the user's own locally generated index; never load untrusted index files.
    return FAISS.load_local(
        str(settings.index_dir),
        _create_embeddings(settings),
        allow_dangerous_deserialization=True,
    )


def get_or_build_index(settings: Settings, rebuild: bool = False) -> FAISS:
    """Reuse an existing index unless rebuilding was requested or it is missing."""
    if not rebuild and index_exists(settings.index_dir):
        logger.info("Reusing saved FAISS index at %s.", settings.index_dir)
        return load_index(settings)

    if not rebuild and settings.index_dir.exists():
        logger.warning("Saved index is incomplete; rebuilding it from the PDF files.")

    documents = load_pdf_documents(settings.data_dir)
    chunks = split_documents(documents, settings)
    return build_index(chunks, settings)
