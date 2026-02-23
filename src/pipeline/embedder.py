"""
Embedding and ChromaDB indexing for fashion article chunks.

Embeds text chunks using sentence-transformers and stores them
in a persistent ChromaDB collection with metadata.
"""

import logging
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_COLLECTION = "fashion_articles"
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"


def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Load a sentence-transformer embedding model.

    Args:
        model_name: HuggingFace model ID. Defaults to all-mpnet-base-v2.
    """
    model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def get_chroma_client(persist_dir: str | Path | None = None) -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    persist_dir = Path(persist_dir or os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR))
    persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"ChromaDB persist directory: {persist_dir}")
    return chromadb.PersistentClient(path=str(persist_dir))


def index_chunks(
    chunks: list[Chunk],
    model: SentenceTransformer | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str | Path | None = None,
    batch_size: int = 64,
) -> chromadb.Collection:
    """Embed chunks and index them into ChromaDB.

    Args:
        chunks: List of Chunk objects to index.
        model: Sentence-transformer model (loaded if not provided).
        collection_name: Name of the ChromaDB collection.
        persist_dir: Directory for ChromaDB persistence.
        batch_size: Number of chunks to embed at once.

    Returns:
        The ChromaDB collection with indexed chunks.
    """
    if not chunks:
        raise ValueError("No chunks to index")

    if model is None:
        model = get_embedding_model()

    client = get_chroma_client(persist_dir)

    # Delete existing collection if it exists (fresh index)
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Process in batches
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = ids[start:end]

        # Generate embeddings
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

        # Convert metadata values to strings (ChromaDB requirement)
        clean_metadatas = []
        for m in batch_metadatas:
            clean_metadatas.append({k: str(v) for k, v in m.items()})

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=clean_metadatas,
            ids=batch_ids,
        )

        logger.info(f"  Indexed batch {start}-{end} of {len(chunks)}")

    logger.info(f"Indexed {len(chunks)} chunks into collection '{collection_name}'")
    return collection


def get_collection(
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str | Path | None = None,
) -> chromadb.Collection:
    """Get an existing ChromaDB collection."""
    client = get_chroma_client(persist_dir)
    return client.get_collection(name=collection_name)


if __name__ == "__main__":
    from src.pipeline.chunker import chunk_articles
    from src.scraper.scrape import load_articles

    articles = load_articles()
    if not articles:
        print("No articles found. Run the scraper first.")
    else:
        chunks = chunk_articles(articles)
        model = get_embedding_model()
        collection = index_chunks(chunks, model=model)
        print(f"Indexed {collection.count()} chunks")
