"""Load PDF documents while retaining source and page metadata."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_documents(data_dir: Path) -> list[Document]:
    """Load every PDF in a directory or raise a clear error if none are present."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"PDF data folder does not exist: {data_dir}. Create it and add PDF files."
        )

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Add at least one PDF and try again."
        )

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        for document in PyPDFLoader(str(pdf_path)).load():
            document.metadata["source"] = pdf_path.name
            documents.append(document)

    logger.info("Loaded %d PDF file(s), totaling %d page(s).", len(pdf_paths), len(documents))
    return documents
