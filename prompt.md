You are a senior Python engineer. Build a small, clean, production-style proof of concept (PoC) for a RAG (Retrieval-Augmented Generation) document Q&A assistant. Everything must run 100% free and locally: Hugging Face embeddings, FAISS, and an Ollama LLM. No paid API keys.

## What the app does
1. Load all PDF files from the ./data folder (read the path from .env).
2. Extract the text from each PDF and split it into small chunks.
3. Turn each chunk into embeddings using Hugging Face (sentence-transformers/all-MiniLM-L6-v2).
4. Store the embeddings in a FAISS index and save it to disk, so it is not rebuilt on every run.
5. When a user asks a question, embed the question, find the top-k closest chunks in FAISS, and send them with the question to an Ollama LLM using LangChain.
6. Print a short, clear answer plus source citations (file name and page number).

## Tech stack
- Python 3.10+
- LangChain (langchain, langchain-community, langchain-text-splitters, langchain-huggingface, langchain-ollama)
- FAISS (faiss-cpu), sentence-transformers, pypdf, python-dotenv
- LLM: Ollama (default model: llama3.2), run locally

## Project structure
```
rag-poc/
├── data/                  # put PDF files here (add a .gitkeep)
├── faiss_index/           # auto-created, git-ignored
├── src/
│   ├── config.py          # loads settings from .env
│   ├── loader.py          # load all PDFs from DATA_DIR
│   ├── chunker.py         # split documents into chunks
│   ├── vector_store.py    # build / save / load FAISS with HF embeddings
│   ├── rag_chain.py       # retriever + prompt + Ollama LLM, returns answer + sources
│   └── main.py            # CLI entry point
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements for each part
- config.py: use python-dotenv. Settings: DATA_DIR, INDEX_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE. Provide sensible defaults.
- loader.py: use PyPDFLoader, scan DATA_DIR for *.pdf files. Raise a clear error if the folder is empty or missing. Keep file name and page number in metadata. Print how many files and pages were loaded.
- chunker.py: use RecursiveCharacterTextSplitter with chunk size and overlap from config (default 800 and 100).
- vector_store.py: use HuggingFaceEmbeddings. Add `build_index()` and `load_index()`. Save with `save_local` and load with `load_local` (allow_dangerous_deserialization=True, with a short comment explaining why it is safe for a local index). Reuse the saved index if it exists. Support a `--rebuild` flag to force rebuilding.
- rag_chain.py: use the ChatOllama model. Use a strict prompt: answer ONLY from the given context, keep the answer short and simple, and reply exactly "I don't know based on the provided documents." if the answer is not in the context. Return the answer and a de-duplicated list of sources (file name and page, shown as page number starting from 1).
- main.py: use argparse with these options:
  - `--rebuild` rebuild the FAISS index
  - `--question "..."` ask one question and exit
  - with no `--question`, start an interactive loop (type `exit` to quit)
  Print the answer, then "Sources:" with file and page. Add `--show-chunks` to also print retrieved chunks with similarity scores for debugging.
- Handle errors well: Ollama not running or the model not pulled (show a helpful message with the fix), empty data folder, and missing index.
- Add logging, type hints, and short docstrings. Keep the code simple and readable.

## .env.example
Include all config keys with default values and one-line comments, for example:
DATA_DIR=./data
INDEX_DIR=./faiss_index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=4
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0

## requirements.txt
List all dependencies with sensible minimum versions. No OpenAI packages.

## .gitignore
Ignore .env, faiss_index/, __pycache__/, .venv/, *.pyc, and data/*.pdf (keep data/.gitkeep).

## README.md
Include:
1. Project overview and a simple ASCII flow diagram: PDFs -> chunks -> embeddings -> FAISS -> retriever -> Ollama -> answer + sources
2. Prerequisites: Python 3.10+, Ollama installed
3. Setup steps: create a virtual environment, install requirements, copy .env.example to .env, run `ollama pull llama3.2`, start Ollama
4. How to add PDFs to ./data and build the index
5. Usage examples for all CLI options
6. Example questions and sample output with sources
7. Configuration table explaining every .env setting
8. How it works, step by step, in simple English
9. Troubleshooting: Ollama connection refused, model not found, slow first run (the embedding model downloads once), empty answers, and what to do if you change the embedding model (delete faiss_index and rebuild)
10. Limitations and next steps: metadata filtering, re-ranking, a managed vector store, and a simple UI

## Output
Give the complete code for every file, with the file path as a heading above each code block. Do not skip any file. Make sure the project runs end to end with: `python -m src.main --rebuild` followed by `python -m src.main --question "How many days of annual leave do I get?"`.