"""
Retrieval module for the fashion stylist RAG pipeline.

Handles query embedding, similarity search via ChromaDB, and
optional cross-encoder reranking for improved precision.
"""

import logging
from dataclasses import dataclass

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.pipeline.embedder import (
    DEFAULT_COLLECTION,
    get_chroma_client,
    get_embedding_model,
)

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its relevance score and metadata."""

    text: str
    score: float
    metadata: dict

    @property
    def source_url(self) -> str:
        return self.metadata.get("source_url", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def site(self) -> str:
        return self.metadata.get("site", "")


class Retriever:
    """Retrieves relevant fashion article chunks for a user query.

    Supports optional cross-encoder reranking for improved precision.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir: str | None = None,
        use_reranker: bool = True,
    ):
        self.embedding_model = embedding_model or get_embedding_model()
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.reranker = None

        if use_reranker:
            try:
                logger.info(f"Loading reranker: {RERANKER_MODEL}")
                self.reranker = CrossEncoder(RERANKER_MODEL)
            except Exception as e:
                logger.warning(f"Could not load reranker, proceeding without: {e}")

    def _get_collection(self) -> chromadb.Collection:
        """Get the ChromaDB collection."""
        client = get_chroma_client(self.persist_dir)
        return client.get_collection(name=self.collection_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        site_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User's styling question or wardrobe description.
            top_k: Number of final chunks to return.
            candidate_k: Number of candidates to fetch before reranking.
                Used only when reranker is enabled.
            site_filter: Optional filter to only return chunks from a
                specific site (e.g., "gq").

        Returns:
            List of RetrievedChunk objects, sorted by relevance.
        """
        collection = self._get_collection()

        # Embed the query
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build ChromaDB query
        fetch_k = candidate_k if self.reranker else top_k
        where_filter = {"site": site_filter} if site_filter else None

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            logger.warning("No results found for query")
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Convert cosine distances to similarity scores
        chunks = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1.0 - dist  # cosine distance -> similarity
            chunks.append(RetrievedChunk(text=doc, score=score, metadata=meta))

        # Rerank if reranker is available
        if self.reranker and len(chunks) > top_k:
            chunks = self._rerank(query, chunks, top_k)
        else:
            chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]

        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
        return chunks

    def _rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Rerank chunks using a cross-encoder model."""
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
        logger.info(f"Reranked {len(chunks)} candidates down to {top_k}")
        return reranked
