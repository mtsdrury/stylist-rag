# Fashion Stylist RAG Assistant

A conversational fashion styling assistant powered by Retrieval-Augmented Generation (RAG). Describe what you're wearing or what's in your closet, and get grounded outfit advice backed by real fashion editorial content.

## Architecture

```
User Query
    |
    v
[Embedding Model] ──> sentence-transformers (all-mpnet-base-v2)
    |
    v
[Vector Search] ──> ChromaDB (cosine similarity, metadata filtering)
    |
    v
[Cross-Encoder Reranker] ──> ms-marco-MiniLM-L-6-v2 (optional)
    |
    v
[LLM Generation] ──> Mistral / GPT-4o-mini (grounded prompt with citations)
    |
    v
[Streamlit UI] ──> Chat interface with source display
```

### Pipeline Overview

1. **Scraping**: Collects fashion editorial articles from GQ, Highsnobiety, Put This On, and The Fashionisto using `requests` + `BeautifulSoup4`.
2. **Chunking**: Splits articles into overlapping text segments (~500 chars, 50 char overlap) using LangChain's `RecursiveCharacterTextSplitter`, preserving metadata (source URL, title, site, date).
3. **Embedding**: Encodes chunks with `sentence-transformers` (`all-mpnet-base-v2`) for dense vector representations.
4. **Indexing**: Stores embeddings in ChromaDB with metadata for filtered retrieval by source site, category, or date.
5. **Retrieval**: Embeds user queries with the same model, performs cosine similarity search, and optionally reranks candidates with a cross-encoder.
6. **Generation**: Constructs a grounded prompt with retrieved context and generates styling advice via OpenAI or HuggingFace Inference API. The LLM cites source articles in its response.

## Setup

```bash
# Clone the repo
git clone https://github.com/mtsdrury/fashion-stylist-rag.git
cd fashion-stylist-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### 1. Scrape articles

```bash
python -m src.scraper.scrape --sites gq putthison thefashionisto --max-articles 50
```

### 2. Build the index

```bash
python -m src.pipeline.embedder
```

### 3. Launch the app

```bash
streamlit run src/app.py
```

### 4. Evaluate retrieval quality

```bash
python -m src.evaluation.evaluate
```

## Example Queries

- "I have a navy blazer and brown chelsea boots, what pants should I wear?"
- "How do I style wide-leg trousers for spring?"
- "What goes with a black leather jacket for a casual date?"
- "Best sneakers for smart casual outfits?"

## Project Structure

```
fashion-stylist-rag/
├── src/
│   ├── scraper/          # Web scraping (sources.py, scrape.py)
│   ├── pipeline/         # Chunking, embedding, retrieval
│   ├── generation/       # LLM prompt construction + generation
│   ├── evaluation/       # Retrieval and answer quality metrics
│   └── app.py            # Streamlit chat interface
├── data/
│   ├── raw/              # Scraped articles (JSON)
│   └── chroma_db/        # Persisted vector store
├── tests/                # Unit tests
└── .github/workflows/    # CI (lint + test)
```

## Tech Stack

- **Embeddings**: sentence-transformers (all-mpnet-base-v2)
- **Vector Store**: ChromaDB (persistent, metadata-filtered)
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**: HuggingFace Inference API (Mistral) or OpenAI (GPT-4o-mini)
- **Chunking**: LangChain RecursiveCharacterTextSplitter
- **Scraping**: requests + BeautifulSoup4
- **UI**: Streamlit
- **Evaluation**: Keyword-based retrieval precision/recall/MRR, n-gram faithfulness scoring

## Evaluation Metrics

| Metric | Description |
|---|---|
| Precision | Fraction of retrieved chunks that are relevant to the query |
| Recall | Fraction of expected keywords found in retrieved chunks |
| MRR | Mean Reciprocal Rank of the first relevant result |
| Faithfulness | N-gram overlap between the generated answer and source context |

## License

MIT
