"""Command-line entry point for the local PDF RAG assistant."""

import argparse
import logging

from src.config import load_settings
from src.rag_chain import QAResult, answer_question
from src.vector_store import get_or_build_index

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Ask questions about local PDF documents.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index from PDFs.")
    parser.add_argument("--question", help="Ask one question, then exit.")
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print retrieved passages and FAISS distance scores.",
    )
    return parser.parse_args()


def _print_result(result: QAResult, show_chunks: bool) -> None:
    """Print an answer, its source citations, and optional retrieval details."""
    print(f"\nAnswer: {result.answer}")
    print("Sources:")
    if result.sources:
        for filename, page in result.sources:
            print(f"- {filename}, page {page}")
    else:
        print("- None")

    if show_chunks:
        print("\nRetrieved chunks (lower FAISS distance is closer):")
        for item in result.chunks:
            filename, page = item.document.metadata.get("source", "Unknown source"), int(
                item.document.metadata.get("page", 0)
            ) + 1
            print(f"\n[{filename}, page {page}; score={item.score:.4f}]")
            print(item.document.page_content)


def main() -> None:
    """Build or load the index and run one-shot or interactive Q&A."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    try:
        settings = load_settings()
        vector_store = get_or_build_index(settings, rebuild=args.rebuild)

        if args.question:
            result = answer_question(
                vector_store, args.question, settings, show_chunks=args.show_chunks
            )
            _print_result(result, args.show_chunks)
            return

        print("Ask a question about your PDFs. Type 'exit' to quit.")
        while True:
            try:
                question = input("\nQuestion: ").strip()
            except EOFError:
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            result = answer_question(
                vector_store, question, settings, show_chunks=args.show_chunks
            )
            _print_result(result, args.show_chunks)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
