How to Run

Train
python -m scripts.train

Build RAG index
rm -rf artifacts
mkdir -p artifacts/chroma
python -m src.rag_copilot.ingest --ticker AAPL --docs_dir data/kb --persist_dir artifacts/chroma --collection_name finance_copilot

Serve API (Ollama)
export RAG_PROVIDER=ollama
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.1
export RAG_PERSIST_DIR=$(pwd)/artifacts/chroma
export RAG_COLLECTION_NAME=finance_copilot
export HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
python -m uvicorn app:app --host 127.0.0.1 --port 8000
