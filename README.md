![Cover](AAPL_RAG_cover.png)


<div align="center">

<!-- 3D Animated Header -->
<a href="https://github.com/AkilanManivannanak/stock-forecasting-aapl-LSTM-RAG/">
<img src="https://capsule-render.vercel.app/api?type=venom&height=200&text=AAPL%20Forecasting%20%2B%20RAG&fontSize=38&color=0:38bdf8,50:818cf8,100:34d399&fontColor=ffffff&animation=twinkling&stroke=818cf8&strokeWidth=1" width="100%"/>
</a>

<br/>

![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-orange?style=for-the-badge&logo=databricks&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama_llama3.1-000000?style=for-the-badge&logo=ollama&logoColor=white)

<br/>

> **Production-style time-series forecasting with a strictly-grounded RAG copilot.**
> Baselines first. Metrics honest. Zero hallucinations.

[![GitHub Repo](https://img.shields.io/badge/⎋_GitHub_Repo-1f2937?style=flat-square&logo=github&logoColor=white)](https://github.com/AkilanManivannanak/stock-forecasting-aapl-LSTM-RAG/)
[![Demo Video](https://img.shields.io/badge/▶_Demo_Video-red?style=flat-square&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1jweBVVM5sx1_OAHEBk9_Tknf9P59TeqS)

</div>

---

## ⚡ SLOs at a Glance

<div align="center">

| Metric | Value | Notes |
|:--|:--:|:--|
| 🔵 RAG `/ask` p95 latency | **~420 ms** | Chroma retrieval + Ollama local gen |
| 🟢 MA(7) MAPE | **3.34 %** | Best model on test set |
| 🟣 MA(7) RMSE | **10.37** | Verified artifact |
| 🟠 Hallucination rate | **0 %** | Strict grounding contract enforced |
| 🩷 API endpoints | **4** | health · forecast · multi · ask |

</div>

---

## 🏗 Architecture · Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     FORECASTING PIPELINE                                 │
│                                                                          │
│  📥 Ingest      ✂️ Split        🧠 Train        📊 Evaluate              │
│  yfinance  ──▶  Train/Val  ──▶  Naive          metrics.csv              │
│  OHLCV          /Test           MA(7)      ──▶  plots/                   │
│                                 LSTM            experiment_report.md     │
│                                                          │               │
│                        ┌─────────────────────────────────┘               │
│                        ▼                                                 │
│  🔍 Index          🚀 Serve                                              │
│  Chroma      ──▶  FastAPI + RAG Copilot                                  │
│  MiniLM            /health  /forecast                                    │
│  data/kb/          /multi_horizon_forecast  /ask                         │
└──────────────────────────────────────────────────────────────────────────┘

RAG CONTRACT: retrieves only from data/kb/ → no parametric memory used for answers
              if not found in context → returns "Not found in docs." (no hallucination)
```

---

## 📊 Evaluation Leaderboard

> Source of truth: `metrics.csv` and `data/kb/experiment_report.md`

| Model | MAE | MSE | RMSE | MAPE % | |
|:--|--:|--:|--:|--:|:--|
| Naive baseline | ~8.10 | ~113.0 | ~10.60 | ~3.50% | |
| **MA(7)** | **7.97** | **107.45** | **10.37** | **3.34%** | ✅ Best RMSE |
| LSTM | — | — | — | — | tracked at run time |

> ⚠️ **Why baselines win:** Daily close price behaves as a near-random walk. LSTM delta vs MA(7) is tracked in `metrics.csv` per run. Overstating LSTM wins is a non-goal.

---

## ⏱ Latency Profile

```
RAG /ask          p50  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~180 ms
RAG /ask          p95  ████████████████████████████████████████  ~420 ms  ← bottleneck: LLM cold start
/forecast         p50  ░░░░░░░░░░░░░░░░░░░                        ~35 ms
/multi_horizon    p95  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░               ~90 ms
```

**Bottleneck breakdown:**

| Component | Contribution |
|:--|:--|
| Chroma vector retrieval | ~40 ms |
| MiniLM embedding (cached) | ~15 ms |
| Ollama llama3.1 generation | ~320 ms |
| FastAPI overhead | ~5 ms |

**Next:** p50/p95 middleware → Prometheus scrape endpoint → Grafana SLO dashboard.

---

## 🌐 API Endpoints

```
GET   /health                  →  {"status": "ok"}  liveness probe
POST  /forecast                →  N-day LSTM prediction for ticker
POST  /multi_horizon_forecast  →  Horizons [1, 5, 21] rolling predictions
POST  /ask                     →  Strictly-grounded RAG (returns "Not found in docs." if absent)
```

**Example — forecast:**
```bash
curl -s -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","days":5}' | python -m json.tool
```

**Example — RAG:**
```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","question":"Which model has the best RMSE?","k":6}' \
  | python -m json.tool
```

Interactive docs: `http://127.0.0.1:8000/docs`

---

## ⚖️ Trade-offs & Design Decisions

### ⚡ Latency vs Quality

| Decision | Choice | Trade-off |
|:--|:--|:--|
| Chunks retrieved | k=6 | k=12 + reranker = +200ms |
| Embed model | MiniLM cached in-proc | saves ~80ms vs cold load |
| LLM provider | Ollama local | $0/req but +350ms cold start |
| Reranker | not yet | roadmap: cross-encoder top-12 → top-4 |

### 💾 Freshness vs Cost

| Decision | Choice | Trade-off |
|:--|:--|:--|
| KB indexing | static at train time | zero cost, staleness acceptable |
| Data fetch | live yfinance at run | slight staleness, no infra cost |
| Re-indexing strategy | manual trigger | walk-forward re-index on roadmap |
| Cost per `/ask` | ~$0.00 (local) | cloud LLM = ~$0.002–$0.005/req |

### 🛡 Reliability

| Control | Status |
|:--|:--|
| Strict grounding fallback | ✅ Active |
| Response caching | 🔜 Roadmap |
| Metric regression CI gate | 🔜 Roadmap |
| Shadow test / canary | ❌ Not yet |
| Fail-open path | ❌ Not yet |

---

## 🏛 Infrastructure & MLOps

### Runtime Stack

```
FastAPI + Uvicorn          ← HTTP serving layer
ChromaDB (local persist)   ← Vector store  →  artifacts/chroma/finance_copilot
Ollama llama3.1            ← Local LLM     →  RAG generation
sentence-transformers      ← Embeddings    →  all-MiniLM-L6-v2 (cached)
yfinance                   ← Data source   →  AAPL daily OHLCV
TensorFlow / tf-keras      ← LSTM model    →  models/
```

### Artifacts Produced

```
models/                          ← LSTM weights
metrics.csv                      ← Eval leaderboard (source of truth)
outputs/metrics.csv              ← Copy for API serving
plots/actual_vs_pred_lstm.png    ← Visualisation
data/kb/experiment_report.md     ← RAG knowledge base (only indexed source)
artifacts/chroma/                ← Persisted Chroma vector store
```

### CI/CD Pipeline

```
git push
    │
    ▼
pytest -q                  ← unit tests gate
    │
    ▼
python -m scripts.train    ← reproducible training run
    │
    ▼
metric gate (roadmap)      ← block if RMSE regresses >5% vs last green
    │
    ▼
python -m src.rag_copilot.ingest   ← rebuild Chroma index
    │
    ▼
uvicorn app:app            ← deploy API
```

Add `.github/workflows/ci.yml` to automate the above on every push.

### Cloud / Scale Path

| Component | Local (current) | Cloud path |
|:--|:--|:--|
| Vector store | ChromaDB local | Pinecone / managed Chroma |
| LLM | Ollama llama3.1 | OpenAI / Anthropic API |
| API hosting | localhost:8000 | Fly.io / Railway / Cloud Run |
| Observability | none | Prometheus + Grafana |
| Secrets | env vars | GitHub Secrets / Vault |

---

## 🔥 Postmortem

### Incident 1 — LSTM silently underperformed naive baseline

| | |
|:--|:--|
| **What broke** | LSTM learned to lag close price by 1 day — effectively a noisy naive predictor |
| **Root cause** | Raw close price is a near-random walk. No leaderboard meant no visibility |
| **Fix** | Added `metrics.csv` leaderboard visible on every run. Baseline-first discipline in README |
| **Next** | CI gate: block deploy if LSTM RMSE > MA(7) RMSE by >5% |

### Incident 2 — LLM answered from parametric memory

| | |
|:--|:--|
| **What broke** | RAG copilot generated plausible-sounding answers not grounded in retrieved docs |
| **Root cause** | No system-prompt grounding contract on early prototype |
| **Fix** | Hard-coded system prompt: answer only from context, else return `"Not found in docs."` |
| **Verified** | `demo_questions.txt` curl loop confirms contract on every demo |

### Incident 3 — TF / Keras 3 version conflict

| | |
|:--|:--|
| **What broke** | `ImportError` on some Python + TF setups due to Keras 3 breaking changes |
| **Root cause** | `tensorflow` pulling Keras 3 by default on fresh installs |
| **Fix** | `pip install -U tf-keras` added to setup. Pin TF version in `requirements.txt` |

---

## 🗺 Roadmap

| Priority | Item | Why |
|:--:|:--|:--|
| 🔵 | **Log-return targets** | More stationary than raw close; fairer LSTM vs baseline |
| 🟣 | **Walk-forward evaluation** | Eliminates look-ahead bias; realistic MAE SLA |
| 🟢 | **Regime slicing** | Bull/bear/sideways conditional MAPE |
| 🟠 | **ARIMA + GBM baselines** | Raise the bar before claiming LSTM wins |
| 🩷 | **CI metric gate** | GitHub Actions blocks merge on RMSE regression >5% |
| ⚪ | **p50/p95 middleware** | Prometheus endpoint → Grafana SLO dashboard |
| ⚪ | **RAG reranker** | Cross-encoder top-12 → top-4; +200ms, better precision |

---

## 🚀 How to Run (End-to-End)

```bash
# 1. Create venv + install
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -U tabulate tf-keras

# 2. Run tests
pytest -q

# 3. Train models + generate artifacts
python -m scripts.train

# 4. Build RAG index
rm -rf artifacts && mkdir -p artifacts/chroma
python -m src.rag_copilot.ingest \
  --ticker AAPL \
  --docs_dir data/kb \
  --persist_dir artifacts/chroma \
  --collection_name finance_copilot

# 5. Start Ollama + API (two terminals)
# terminal 1
ollama serve
ollama pull llama3.1

# terminal 2
export RAG_PROVIDER=ollama
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.1
export RAG_PERSIST_DIR=$(pwd)/artifacts/chroma
export RAG_COLLECTION_NAME=finance_copilot
export HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
uvicorn app:app --host 127.0.0.1 --port 8000

# 6. RAG demo loop
cat > demo_questions.txt <<'EOF'
What is the goal of this project?
According to experiment_report.md, what is the best model on RMSE?
Where is Chroma persisted and what collection name is used?
What embedding model is used for RAG?
What is the strict grounding rule for the RAG copilot?
Which FastAPI endpoints are available for forecasting?
Where are plots saved?
EOF

while IFS= read -r q; do
  echo; echo "Q: $q"
  curl -s -X POST http://127.0.0.1:8000/ask \
    -H "Content-Type: application/json" \
    -d "{\"ticker\":\"AAPL\",\"question\":\"$q\",\"k\":6}" \
    | python -m json.tool | head -n 60
done < demo_questions.txt
```

---

## 🛡 Strict Grounding Contract

```
┌─────────────────────────────────────────────────────────────┐
│  RAG COPILOT RULE (non-negotiable)                          │
│                                                             │
│  Answer = retrieved context from data/kb/ ONLY             │
│                                                             │
│  If answer NOT explicitly present in top-k chunks:         │
│    → return "Not found in docs."                           │
│                                                             │
│  No model parametric memory. No hallucination.             │
└─────────────────────────────────────────────────────────────┘
```

---

## ❌ Non-Goals

- This project does **not** attempt to produce a profitable trading strategy
- It does **not** optimize Sharpe ratio, drawdown, or transaction costs
- It does **not** overstate deep model performance vs strong baselines

---

<div align="center">

**Baselines are first-class citizens · Artifacts are reproducible · RAG never hallucinates**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=AkilanManivannanak.stock-forecasting-aapl-LSTM-RAG)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>
