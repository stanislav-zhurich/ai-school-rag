# DJT Tweet Analyzer

A **Retrieval-Augmented Generation (RAG)** application that lets you explore and query Donald Trump's tweets (2009–2025) through a Streamlit web interface. The app combines semantic search over a vector database with an Azure OpenAI language model to answer natural-language questions about the tweet corpus.

**Dataset:** [Trump Tweets 2009–2025](https://www.kaggle.com/datasets/datadrivendecision/trump-tweets-2009-2025) on Kaggle.

---

## 1. What the Application Does

- **Ingests** up to tens of thousands of tweets from the [Kaggle dataset](https://www.kaggle.com/datasets/datadrivendecision/trump-tweets-2009-2025) into a ChromaDB vector store (run once).
- **Answers questions** in natural language using RAG: the query is embedded, the most relevant tweet chunks are retrieved, and an LLM synthesises a grounded answer.
- **Visualises** the tweet corpus across time, platforms, engagement, hashtags, and post types.
- **Evaluates** the RAG pipeline on a built-in set of 10 reference questions using [RAGAS](https://docs.ragas.io/) metrics (Faithfulness, Context Precision, Context Recall).

---

## 2. Application Structure

```
ai-school-rag/
├── src/
│   ├── app.py                    # Streamlit entry point (tab wiring only)
│   ├── main.py                   # One-time ingestion entry point
│   ├── RAG.py                    # RAG orchestrator (embed → retrieve → generate)
│   ├── config.py                 # Dataclass-based settings (Azure + Pipeline)
│   │
│   ├── ui/                       # Streamlit UI package
│   │   ├── __init__.py           # Re-exports renderers and cache factories
│   │   ├── constants.py          # Year/platform options, quality thresholds
│   │   ├── cache.py              # @st.cache_data / @st.cache_resource factories
│   │   ├── filters.py            # build_where_filter() for ChromaDB queries
│   │   ├── dashboard_tab.py      # Dashboard tab renderer
│   │   ├── rag_tab.py            # RAG Assistant tab renderer
│   │   └── eval_tab.py           # Evaluation tab renderer
│   │
│   ├── ingestion/
│   │   └── pipeline.py           # IngestionPipeline: load → chunk → embed → store
│   │
│   ├── loaders/
│   │   └── csv_loader.py         # Parses and caches tweets from CSV
│   │
│   ├── chunkers/
│   │   ├── base.py               # BaseChunker abstract class
│   │   ├── factory.py            # ChunkerFactory (strategy pattern)
│   │   ├── identity.py           # One chunk per tweet
│   │   ├── sliding_window.py     # Fixed-size overlapping windows
│   │   └── semantic.py           # Embedding-similarity grouping
│   │
│   ├── embedder/
│   │   ├── base_embedder.py      # BaseEmbedder (batching, retry, fallback)
│   │   ├── nomic_embedder.py     # Local Nomic model via LM Studio
│   │   └── openai_embedder.py    # Azure OpenAI embeddings (DIAL proxy)
│   │
│   ├── vectorstore/
│   │   └── chromadb_store.py     # ChromaDB persistence + cosine search
│   │
│   ├── model/
│   │   ├── tweet.py              # Tweet dataclass
│   │   ├── chunk.py              # Chunk dataclass
│   │   └── search_result.py      # SearchResult dataclass
│   │
│   └── eval/
│       ├── eval_dataset.py       # 10 reference questions + reference answers
│       └── evaluator.py          # RAGAS-based evaluation (Faithfulness, Precision, Recall)
│
├── data/
│   ├── raw/                      # Downloaded Kaggle CSV (auto-created)
│   └── processed/                # Parsed tweet cache (auto-created)
│
├── chroma_db_identity/           # ChromaDB persistent store (auto-created)
├── pyproject.toml
└── .env                          # API keys (not committed)
```

---

## 3. Architecture

### Ingestion (run once)

```
Kaggle CSV
    │
    ▼
CSVLoader ──► Tweet[]
    │
    ▼
ChunkerFactory
(identity | sliding_window | semantic)
    │
    ▼
Chunk[]  ──► BaseEmbedder.embed_chunks() ──► float[][]
                                                 │
                                                 ▼
                                         ChromaDBStore.add_chunks()
                                         (persistent cosine index)
```

### Query time (Streamlit app)

```
User question
    │
    ▼
BaseEmbedder.embed_query()
    │
    ▼
ChromaDBStore.search()  ──► top-N SearchResult[]
    │                              │
    ▼                              ▼
Context string            Relevance scores displayed in UI
    │
    ▼
Azure OpenAI Chat (gpt-4o)
    │
    ▼
Grounded answer
```

### Evaluation

```
EvalQuestion[] (10 reference Q&A pairs)
    │
    ▼
RAG.get_answer()  ──► answer + retrieved contexts
    │
    ▼
RAGAS
  ├── Faithfulness           (no hallucination)
  ├── LLM Context Precision  (retrieval relevance)
  └── LLM Context Recall     (retrieval completeness)
    │
    ▼
Per-question scores + aggregate metrics displayed in UI
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Ingestion and query are separate processes | Ingestion is expensive (API calls, large data); the app only reads the pre-built index |
| `BaseEmbedder` with pluggable backends | Swap between local Nomic (free) and Azure OpenAI without changing any other code |
| `ChunkerFactory` (strategy pattern) | Change chunking strategy via a single config value |
| RAGAS LLM-as-judge | Quantitative evaluation without manually labelled answers for every metric |

---

## 4. How to Run

### Prerequisites

- Python 3.10–3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- Azure OpenAI access (API key for the EPAM DIAL proxy)

### Setup

```powershell
# 1. Clone the repository and enter the project directory
cd ai-school-rag

# 2. Install dependencies
poetry install

# 3. Create a .env file with your API key
echo "OPENAI_API_KEY=<your-azure-openai-key>" > .env
```

### Step 1 — Build the vector index (run once)

```powershell
poetry run python src/main.py
```

This downloads the dataset, chunks the tweets, computes embeddings, and populates ChromaDB.

The vector store directory is named after the active chunking strategy — `chroma_db_<strategy>`:

| `CHUNKING_STRATEGY` | Directory |
|---|---|
| `"identity"` | `chroma_db_identity/` |
| `"sliding_window"` | `chroma_db_sliding_window/` |
| `"semantic"` | `chroma_db_semantic/` |

Each strategy gets its own isolated store, so you can build and compare multiple indexes without overwriting each other. The app automatically connects to the directory that matches the current `CHUNKING_STRATEGY` value in `config.py`.

Subsequent runs skip ingestion automatically if the target store already contains data.

To rebuild from scratch (e.g. after changing tweet filters or the embedding model):

```powershell
# Delete the store for the active strategy and the processed tweet cache
Remove-Item -Recurse -Force chroma_db_identity, data/processed   # adjust name to match strategy
poetry run python src/main.py
```

### Step 2 — Launch the web application

```powershell
poetry run streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Configuration (`src/config.py`)

Configuration is exposed as two frozen dataclasses plus module-level aliases
for backwards compatibility:

- `config.pipeline` (`PipelineSettings`) — dataset, chunking, vector store
- `config.azure()` (`AzureSettings`) — endpoint, deployment, API key (read
  lazily so the module imports even without `OPENAI_API_KEY` set)

**Pipeline settings**

| Attribute | Default | Description |
|---|---|---|
| `chunking_strategy` | `"identity"` | `"identity"` / `"sliding_window"` / `"semantic"` |
| `max_tweets` | `10_000` | Maximum tweets sampled for ingestion and dashboard |
| `kaggle_dataset_handle` | `"datadrivendecision/trump-tweets-2009-2025"` | Source dataset on Kaggle |
| `collection_name` | `"documents"` | ChromaDB collection name |
| `chroma_path` | `./chroma_db_<strategy>` | Persistent store directory (derived from `chunking_strategy`) |
| `raw_dir` / `processed_dir` | `data/raw` / `data/processed` | Dataset locations |

**Azure settings**

| Attribute | Default | Description |
|---|---|---|
| `endpoint` | `"https://ai-proxy.lab.epam.com"` | Azure OpenAI / DIAL endpoint |
| `chat_deployment` | `"gpt-4o"` | Azure OpenAI chat deployment name |
| `api_version` | `"2024-10-21"` | Azure OpenAI API version |
| `api_key` | *from `OPENAI_API_KEY`* | Read from env on first use |

Legacy module-level constants (`CHUNKING_STRATEGY`, `MAX_TWEETS`,
`COLLECTION_NAME`, `DIAL_URL`, `CHAT_MODEL`, `API_VERSION`, `API_KEY`,
`KAGGLE_DATASET_HANDLE`, `PROCESSING_DIR`) are still exported from
`config` so existing imports keep working.

---

## 5. How to Use — UI Guide

The application has three tabs.

### 📊 Dashboard

Displays analytics over the full tweet corpus loaded at startup.

| Widget | Description |
|---|---|
| **KPI metrics** | Total posts, years covered, platform count, average favorites and reposts |
| **Posts by Platform** | Donut chart — Twitter vs Truth Social breakdown |
| **Posts per Year** | Bar chart — tweet volume by calendar year |
| **Engagement over Time** | Line chart — monthly favorites and reposts trend |
| **Top Hashtags** | Horizontal bar — most-used hashtags |
| **Post Type Breakdown** | Donut — original / repost / quote / deleted split |
| **Top Mentioned Users** | Bar — most frequently @-mentioned accounts |

---

### 💬 RAG Assistant

Ask natural-language questions and receive answers grounded in the tweet corpus.

1. **Filters** (optional) — narrow retrieval by year and/or platform using the dropdowns at the top.
2. **Chunks to retrieve** — slider (1–10) controlling how many tweet chunks are passed as context to the LLM.
3. **Your question** — type a question and press **Enter** or click **Ask**.
4. **Answer** — the LLM's synthesised response, citing or paraphrasing specific tweets.
5. **Context Relevance Score** — cosine similarity between your query and each retrieved chunk. Scores ≥ 0.55 are 🟢 Good; 0.50–0.54 are 🟡 Fair; below 0.50 are 🔴 Low and trigger a warning that the topic may not be covered in the dataset.
6. **Retrieved Context Chunks** — expandable cards showing the exact tweet text and metadata (date, platform) used to generate the answer.

> **Tip:** If you receive "I cannot answer", check the relevance scores — a low score means the topic is likely not covered in the indexed dataset.

---

### 🧪 Evaluation

Measures RAG pipeline quality on 10 predefined reference questions.

1. Click **▶ Run Evaluation** — this runs all 10 questions through the full RAG pipeline and scores each with RAGAS (takes ~1–2 minutes).
2. **Aggregate Results** — four metric tiles: Faithfulness, Context Precision, Context Recall, Combined Score.
3. **Results by Category** — table grouping scores by topic (Foreign Policy, Domestic Politics, Economy, COVID-19).
4. **Per-question Results** — expandable rows showing individual scores, the generated answer, and the retrieved context chunks for each question.

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are the answer's claims supported by the retrieved context? (anti-hallucination) |
| **Context Precision** | Are the retrieved chunks relevant to the question? |
| **Context Recall** | Does the retrieved context contain enough information to answer fully? |
| **Combined Score** | Simple average of the three metrics above |
