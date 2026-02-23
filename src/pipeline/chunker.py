"""
Document chunking for fashion editorial articles.

Takes raw article JSON and splits body text into overlapping chunks
with metadata attached for downstream retrieval.
"""

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with associated metadata."""

    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return {"text": self.text, **self.metadata}


def clean_text(text: str) -> str:
    """Clean article text by removing extra whitespace and artifacts."""
    # Collapse multiple newlines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
    text = "\n\n".join(cleaned_lines)

    # Remove common artifacts
    for artifact in ["Advertisement", "ADVERTISEMENT", "Continue reading", "READ MORE"]:
        text = text.replace(artifact, "")

    return text.strip()


def chunk_article(
    article: dict,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a single article into overlapping text chunks.

    Args:
        article: Dict with keys: title, url, source, body_text, etc.
        chunk_size: Target size per chunk in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of Chunk objects with metadata.
    """
    body = clean_text(article.get("body_text", ""))
    if not body:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    texts = splitter.split_text(body)

    metadata = {
        "source_url": article.get("url", ""),
        "title": article.get("title", ""),
        "site": article.get("source", ""),
        "author": article.get("author", ""),
        "date": article.get("date", ""),
        "category": article.get("category", ""),
    }

    chunks = []
    for i, text in enumerate(texts):
        chunk_metadata = {**metadata, "chunk_index": i, "total_chunks": len(texts)}
        chunks.append(Chunk(text=text, metadata=chunk_metadata))

    return chunks


def chunk_articles(
    articles: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk a list of articles into retrievable text segments.

    Args:
        articles: List of article dicts from the scraper.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Character overlap between chunks.

    Returns:
        List of all Chunk objects across articles.
    """
    all_chunks = []
    for article in articles:
        chunks = chunk_article(article, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(
        f"Chunked {len(articles)} articles into {len(all_chunks)} chunks "
        f"(avg {len(all_chunks) / max(len(articles), 1):.1f} chunks/article)"
    )
    return all_chunks
