"""Focused regression tests for PDF loading, chunking, indexing, and QA."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from httpx import ConnectError
from langchain_core.documents import Document

from src.chunker import split_documents
from src.config import Settings, load_settings
from src.loader import load_pdf_documents
from src.rag_chain import UNKNOWN_ANSWER, answer_question
from src.vector_store import (
    _create_embeddings,
    build_index,
    get_or_build_index,
    load_index,
)


class ConfigurationTests(unittest.TestCase):
    def test_load_settings_reads_hugging_face_token(self) -> None:
        with (
            patch("src.config.load_dotenv"),
            patch.dict("os.environ", {"HF_TOKEN": "test-token"}, clear=True),
        ):
            settings = load_settings()

        self.assertEqual(settings.hf_token, "test-token")


def make_settings(root: Path, **overrides: object) -> Settings:
    values = {
        "data_dir": root / "data",
        "index_dir": root / "faiss_index",
        "embedding_model": "test-embedding-model",
        "hf_token": None,
        "chunk_size": 10,
        "chunk_overlap": 2,
        "top_k": 2,
        "ollama_model": "test-llm",
        "ollama_base_url": "http://localhost:11434",
        "temperature": 0.0,
    }
    values.update(overrides)
    return Settings(**values)


class PDFLoaderTests(unittest.TestCase):
    def test_missing_data_directory_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "folder does not exist"):
                load_pdf_documents(Path(temp_dir) / "missing")

    def test_empty_data_directory_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "No PDF files found"):
                load_pdf_documents(data_dir)

    def test_loads_pages_and_sets_pdf_filename_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "policy.pdf").touch()
            page = Document(page_content="Policy content", metadata={"page": 0})
            with patch("src.loader.PyPDFLoader") as loader:
                loader.return_value.load.return_value = [page]
                result = load_pdf_documents(data_dir)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata, {"page": 0, "source": "policy.pdf"})


class ChunkerTests(unittest.TestCase):
    def test_chunks_respect_size_and_preserve_page_metadata(self) -> None:
        source = Document(
            page_content="0123456789abcdefghij",
            metadata={"source": "policy.pdf", "page": 2},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = split_documents([source], make_settings(Path(temp_dir)))

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= 10 for chunk in chunks))
        self.assertTrue(
            all(chunk.metadata == {"source": "policy.pdf", "page": 2} for chunk in chunks)
        )
        self.assertEqual(chunks[0].page_content[-2:], chunks[1].page_content[:2])


class VectorStoreTests(unittest.TestCase):
    def test_create_embeddings_passes_configured_hugging_face_token(self) -> None:
        settings = make_settings(Path("."), hf_token="test-token")
        with patch("src.vector_store.HuggingFaceEmbeddings") as embeddings:
            _create_embeddings(settings)

        embeddings.assert_called_once_with(
            model_name="test-embedding-model",
            model_kwargs={"token": "test-token"},
        )

    def test_create_embeddings_omits_empty_hugging_face_token(self) -> None:
        settings = make_settings(Path("."), hf_token=None)
        with patch("src.vector_store.HuggingFaceEmbeddings") as embeddings:
            _create_embeddings(settings)

        embeddings.assert_called_once_with(
            model_name="test-embedding-model",
            model_kwargs={},
        )

    def test_build_index_saves_persistent_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = make_settings(Path(temp_dir))
            fake_store = MagicMock()
            document = Document(page_content="policy")
            with (
                patch("src.vector_store._create_embeddings", return_value="embeddings"),
                patch("src.vector_store.FAISS.from_documents", return_value=fake_store) as create,
            ):
                result = build_index([document], settings)

        self.assertIs(result, fake_store)
        create.assert_called_once_with([document], "embeddings")
        fake_store.save_local.assert_called_once_with(str(settings.index_dir))

    def test_build_index_rejects_empty_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "no text chunks"):
                build_index([], make_settings(Path(temp_dir)))

    def test_reuses_saved_index_without_loading_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = make_settings(Path(temp_dir))
            expected_store = object()
            with (
                patch("src.vector_store.index_exists", return_value=True),
                patch("src.vector_store.load_index", return_value=expected_store) as load,
                patch("src.vector_store.load_pdf_documents") as load_pdfs,
            ):
                result = get_or_build_index(settings)

        self.assertIs(result, expected_store)
        load.assert_called_once_with(settings)
        load_pdfs.assert_not_called()

    def test_load_index_requires_both_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "No complete FAISS index"):
                load_index(make_settings(Path(temp_dir)))


class RAGChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = make_settings(Path("."), top_k=2)
        self.documents = [
            Document(
                page_content="Annual leave is 24 days.",
                metadata={"source": "policy.pdf", "page": 0},
            ),
            Document(
                page_content="Carry forward up to 10 days.",
                metadata={"source": "C:\\policies\\policy.pdf", "page": 0},
            ),
        ]
        self.vector_store = MagicMock()
        self.vector_store.similarity_search_with_score.return_value = [
            (self.documents[0], 0.1),
            (self.documents[1], 0.2),
        ]

    def test_answers_from_retrieved_context_and_deduplicates_sources(self) -> None:
        model_response = SimpleNamespace(content="Full-time employees receive 24 days.")
        with patch("src.rag_chain.ChatOllama") as chat:
            chat.return_value.invoke.return_value = model_response
            result = answer_question(
                self.vector_store,
                "How much annual leave?",
                self.settings,
            )

        self.assertEqual(result.answer, "Full-time employees receive 24 days.")
        self.assertEqual(result.sources, [("policy.pdf", 1)])
        self.assertEqual(result.chunks, [])
        self.assertEqual(chat.return_value.invoke.call_count, 1)
        prompt = chat.return_value.invoke.call_args.args[0]
        self.assertIn("Annual leave is 24 days.", prompt)
        self.assertIn("Carry forward up to 10 days.", prompt)

    def test_returns_exact_unknown_when_retrieval_is_empty(self) -> None:
        self.vector_store.similarity_search_with_score.return_value = []
        with patch("src.rag_chain.ChatOllama") as chat:
            result = answer_question(
                self.vector_store, "Unsupported question", self.settings
            )

        self.assertEqual(result.answer, UNKNOWN_ANSWER)
        self.assertEqual(result.sources, [])
        self.assertEqual(result.chunks, [])
        chat.assert_not_called()

    def test_show_chunks_returns_retrieval_scores(self) -> None:
        with patch("src.rag_chain.ChatOllama") as chat:
            chat.return_value.invoke.return_value = SimpleNamespace(content="Answer")
            result = answer_question(
                self.vector_store,
                "Question",
                self.settings,
                show_chunks=True,
            )

        self.assertEqual(len(result.chunks), 2)
        self.assertEqual([item.score for item in result.chunks], [0.1, 0.2])
        self.vector_store.similarity_search_with_score.assert_called_once_with(
            "Question", k=2
        )

    def test_reports_helpful_error_when_ollama_is_unreachable(self) -> None:
        with patch("src.rag_chain.ChatOllama") as chat:
            chat.return_value.invoke.side_effect = ConnectError("connection refused")
            with self.assertRaisesRegex(RuntimeError, "Cannot connect to Ollama"):
                answer_question(self.vector_store, "Question", self.settings)


if __name__ == "__main__":
    unittest.main()
