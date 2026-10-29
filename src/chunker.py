"""Split loaded PDF pages into overlapping text chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings


def split_documents(
    documents: list[Document], settings: Settings
) -> list[Document]:
    """Split pages using the configured chunk size and overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
