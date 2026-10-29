"""Retrieve relevant PDF passages and answer questions with local Ollama."""

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from ollama import ResponseError
from httpx import ConnectError, ConnectTimeout

from src.config import Settings


UNKNOWN_ANSWER = "I don't know based on the provided documents."
PROMPT_TEMPLATE = """You answer questions using only the supplied document context.
Keep the answer short and simple. Do not use outside knowledge.
If the context does not contain the answer, reply exactly:
I don't know based on the provided documents.

Context:
{context}

Question: {question}
Answer:"""


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved passage and its FAISS distance score."""

    document: Document
    score: float


@dataclass(frozen=True)
class QAResult:
    """An answer with deduplicated source citations and optional retrieved chunks."""

    answer: str
    sources: list[tuple[str, int]]
    chunks: list[RetrievedChunk]


def _source_for(document: Document) -> tuple[str, int]:
    """Return a display filename and one-based page number."""
    source = Path(str(document.metadata.get("source", "Unknown source"))).name
    page = document.metadata.get("page", 0)
    try:
        page_number = int(page) + 1
    except (TypeError, ValueError):
        page_number = 1
    return source, page_number


def _friendly_response_error(exc: ResponseError, settings: Settings) -> RuntimeError:
    """Translate Ollama model errors into actionable guidance."""
    message = str(exc)
    lowered = message.lower()
    if any(term in lowered for term in ("not found", "pull model", "model does not exist")):
        guidance = (
            f"Ollama model '{settings.ollama_model}' is unavailable. "
            f"Install it with `ollama pull {settings.ollama_model}` and try again."
        )
    else:
        guidance = f"Ollama request failed: {message}"
    return RuntimeError(guidance)


def answer_question(
    vector_store: FAISS,
    question: str,
    settings: Settings,
    show_chunks: bool = False,
) -> QAResult:
    """Retrieve supporting passages and generate an answer grounded in them."""
    matches = vector_store.similarity_search_with_score(question, k=settings.top_k)
    retrieved = [RetrievedChunk(document=document, score=float(score)) for document, score in matches]

    if not retrieved:
        return QAResult(answer=UNKNOWN_ANSWER, sources=[], chunks=[])

    context = "\n\n".join(chunk.document.page_content for chunk in retrieved)
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )

    try:
        response = model.invoke(PROMPT_TEMPLATE.format(context=context, question=question))
    except (ConnectError, ConnectTimeout) as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Start Ollama (for example, run `ollama serve`) and try again."
        ) from exc
    except ResponseError as exc:
        raise _friendly_response_error(exc, settings) from exc

    answer = str(response.content).strip()
    sources = list(dict.fromkeys(_source_for(chunk.document) for chunk in retrieved))
    return QAResult(
        answer=answer,
        sources=sources,
        chunks=retrieved if show_chunks else [],
    )
