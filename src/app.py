"""
Streamlit chat interface for the Fashion Stylist RAG assistant.

Provides a conversational UI where users describe their wardrobe or ask
styling questions, and receive grounded advice with source citations.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so imports work when run via `streamlit run`
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Load .env file (local dev); Streamlit Cloud uses st.secrets instead
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

import streamlit as st

logger = logging.getLogger(__name__)

QUERY_LOG_PATH = _project_root / "data" / "query_log.jsonl"
SAMPLE_ARTICLES_PATH = _project_root / "data" / "sample" / "sample_articles.json"


def _get_secret(key: str, default: str = "") -> str:
    """Get a config value from env vars or st.secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _ensure_env_from_secrets():
    """Copy st.secrets into env vars so downstream modules can use os.getenv."""
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_API_TOKEN",
                "LLM_PROVIDER", "HF_MODEL_ID", "EMBEDDING_MODEL"]:
        if not os.getenv(key):
            val = _get_secret(key)
            if val:
                os.environ[key] = val


_ensure_env_from_secrets()

from src.generation.stylist import FashionStylist
from src.pipeline.retriever import Retriever


def log_interaction(query: str, answer: str, sources: list[dict], chunks: list):
    """Append a query/response record to the local log file."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": [
            {"text": c.text[:200], "score": round(c.score, 4), "title": c.title, "site": c.site}
            for c in chunks
        ],
    }
    QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _bootstrap_index():
    """Build the ChromaDB index from sample articles if no index exists."""
    from src.pipeline.chunker import chunk_articles
    from src.pipeline.embedder import get_chroma_client, get_embedding_model, index_chunks, DEFAULT_COLLECTION

    client = get_chroma_client()
    try:
        collection = client.get_collection(name=DEFAULT_COLLECTION)
        if collection.count() > 0:
            logger.info(f"Index already exists with {collection.count()} chunks")
            return
    except Exception:
        pass

    if not SAMPLE_ARTICLES_PATH.exists():
        st.error("No article data found. Run the scraper or provide sample articles.")
        st.stop()

    with open(SAMPLE_ARTICLES_PATH, encoding="utf-8") as f:
        articles = json.load(f)

    chunks = chunk_articles(articles)
    model = get_embedding_model()
    collection = index_chunks(chunks, model=model)
    logger.info(f"Bootstrapped index with {collection.count()} chunks from sample data")


# Page config
st.set_page_config(
    page_title="Fashion Stylist Assistant",
    page_icon="👔",
    layout="wide",
)

st.title("Fashion Stylist Assistant")
st.caption("Ask me anything about styling outfits. Powered by RAG over fashion editorial content.")


@st.cache_resource(show_spinner="Building search index from fashion articles...")
def load_retriever():
    """Load the retriever (cached across reruns). Bootstraps index if needed."""
    _bootstrap_index()
    return Retriever(use_reranker=True)


@st.cache_resource
def load_stylist():
    """Load the fashion stylist LLM (cached across reruns)."""
    return FashionStylist()


# Sidebar controls
with st.sidebar:
    st.header("Settings")

    top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)

    site_filter = st.selectbox(
        "Filter by source",
        options=[
            "All",
            "putthison",
            "thefashionisto",
            "dieworkwear",
            "permanentstyle",
            "corporette",
            "wardrobeoxygen",
            "thefashionspot",
            "coveteur",
            "hypebeast",
            "i_d",
            "refinery29",
        ],
        index=0,
    )

    use_reranker = st.checkbox("Use cross-encoder reranking", value=True)

    st.divider()
    st.markdown("**Example queries:**")
    st.markdown(
        "- How do I style a crochet halter top for summer?\n"
        "- What's the move with ballet flats right now?\n"
        "- I have a navy blazer and brown chelsea boots, what pants should I wear?\n"
        "- What are the biggest fashion trends for 2026?"
    )

    st.divider()
    st.markdown(
        "*Built by [MacKenzie Drury](https://github.com/mtsdrury) "
        "as a RAG portfolio project.*"
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"**[{i}]** [{src['title']}]({src['url']}) ({src['site']})")

# Chat input
if prompt := st.chat_input("What are you wearing? Ask me for styling advice..."):
    # Display user message immediately and save to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response with a single status message
    with st.chat_message("assistant"):
        status = st.status("Finding you the best styling advice...", expanded=False)

        with status:
            st.write("Searching fashion articles...")
            retriever = load_retriever()
            filter_site = None if site_filter == "All" else site_filter
            chunks = retriever.retrieve(
                query=prompt,
                top_k=top_k,
                site_filter=filter_site,
            )

            st.write(f"Found {len(chunks)} relevant sources. Generating advice...")
            stylist = load_stylist()
            response = stylist.generate(query=prompt, chunks=chunks)
            status.update(label="Done!", state="complete")

        st.markdown(response.answer)

        if response.sources:
            with st.expander("Sources"):
                for i, src in enumerate(response.sources, 1):
                    st.markdown(f"**[{i}]** [{src['title']}]({src['url']}) ({src['site']})")

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })

        # Log interaction locally
        log_interaction(prompt, response.answer, response.sources, chunks)
