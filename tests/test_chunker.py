"""Tests for the document chunking module."""

from src.pipeline.chunker import Chunk, chunk_article, chunk_articles, clean_text


def test_clean_text_collapses_whitespace():
    text = "Hello\n\n\n\nWorld\n\n\nTest"
    result = clean_text(text)
    assert "Hello" in result
    assert "World" in result
    assert "\n\n\n" not in result


def test_clean_text_removes_ad_artifacts():
    text = "Some content. Advertisement More content."
    result = clean_text(text)
    assert "Advertisement" not in result


def test_chunk_article_basic():
    article = {
        "title": "How to Wear a Navy Blazer",
        "url": "https://example.com/navy-blazer",
        "source": "gq",
        "author": "Test Author",
        "date": "2024-01-15",
        "category": "style guide",
        "body_text": "A navy blazer is one of the most versatile pieces in menswear. "
        * 20,
    }
    chunks = chunk_article(article, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].metadata["title"] == "How to Wear a Navy Blazer"
    assert chunks[0].metadata["site"] == "gq"
    assert chunks[0].metadata["source_url"] == "https://example.com/navy-blazer"


def test_chunk_article_empty_body():
    article = {"title": "Empty", "url": "", "source": "", "body_text": ""}
    chunks = chunk_article(article)
    assert chunks == []


def test_chunk_articles_multiple():
    articles = [
        {
            "title": f"Article {i}",
            "url": f"https://example.com/{i}",
            "source": "test",
            "body_text": f"Content for article {i}. " * 50,
        }
        for i in range(3)
    ]
    chunks = chunk_articles(articles, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 3  # Should produce multiple chunks per article


def test_chunk_metadata_includes_index():
    article = {
        "title": "Test Article",
        "url": "https://example.com/test",
        "source": "test",
        "body_text": "Paragraph about styling tips and fashion advice. " * 30,
    }
    chunks = chunk_article(article, chunk_size=200, chunk_overlap=30)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["total_chunks"] == len(chunks)
