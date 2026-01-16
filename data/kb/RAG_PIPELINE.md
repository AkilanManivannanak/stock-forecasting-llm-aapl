# RAG Pipeline (Local, Grounded)

Ingestion
- Input: files under data/kb/ (Markdown, text, optional PDFs)
- Chunking: RecursiveCharacterTextSplitter
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector store: Chroma persisted under artifacts/chroma
- Collection name: finance_copilot

Answering (Strict Grounding)
- Use ONLY retrieved context
- If missing, output exactly: Not found in docs.
- Return citations (source + snippet) and retrieval_debug (counts, sources)
