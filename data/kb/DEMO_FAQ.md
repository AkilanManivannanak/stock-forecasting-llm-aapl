# Demo FAQ (Grounded RAG)

Q1. What is the goal of this project?
A1. Build a production-style stock forecasting system for AAPL that demonstrates an end-to-end ML workflow: data ingestion, training + evaluation with baselines, artifacts, FastAPI serving, and a strict RAG copilot that answers using only repo docs.

Q2. Which models and baselines are reported in the experiment report?
A2. Models/baselines: naive (random-walk), moving average (ma_5), random_forest (tabular lag baseline), and LSTM.

Q3. What does the experiment report say is the best model on RMSE?
A3. The experiment report explicitly states the best model on RMSE is the naive baseline.

Q4. Where are plots saved and which plot should I show in a demo?
A4. Plots are saved under the plots/ folder. Show plots/actual_vs_pred_lstm.png in the demo.

Q5. What artifacts does training produce and where are they saved?
A5. Training saves: models/ (trained model), outputs/metrics.csv (leaderboard), plots/ (visuals), and data/kb/experiment_report.md (markdown report).

Q6. Where is the Chroma vector store persisted and what is the collection name?
A6. The Chroma vector store is persisted under artifacts/chroma and the collection name is finance_copilot.

Q7. What embedding model is used for RAG?
A7. The embedding model is sentence-transformers/all-MiniLM-L6-v2.

Q8. Which files are indexed for RAG and from which folder?
A8. RAG indexes Markdown/text/PDF files under data/kb/.

Q9. How does the RAG copilot answer questions (high-level pipeline)?
A9. At query time it loads the persisted Chroma collection, retrieves top-k chunks by similarity, and answers using only retrieved context. If the answer is not explicitly in the retrieved context, it must return exactly: Not found in docs.

Q10. What does the /ask endpoint return?
A10. /ask returns JSON containing: answer, citations (sources + snippets), and retrieval_debug (collection_name, persist_dir, chroma_count, docs_returned).
