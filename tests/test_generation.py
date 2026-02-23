"""Tests for the generation module."""

from src.generation.stylist import build_context_block, build_prompt
from src.pipeline.retriever import RetrievedChunk


def _make_chunk(text: str, title: str = "Test Article", url: str = "https://example.com/test"):
    return RetrievedChunk(
        text=text,
        score=0.9,
        metadata={"title": title, "source_url": url, "site": "test"},
    )


def test_build_context_block_single():
    chunks = [_make_chunk("Navy blazers pair well with gray trousers.")]
    block = build_context_block(chunks)
    assert "Navy blazers" in block
    assert "Test Article" in block
    assert "Context 1" in block


def test_build_context_block_empty():
    assert "No relevant context" in build_context_block([])


def test_build_context_block_multiple():
    chunks = [
        _make_chunk("Chunk one content.", title="Article A", url="https://a.com"),
        _make_chunk("Chunk two content.", title="Article B", url="https://b.com"),
    ]
    block = build_context_block(chunks)
    assert "Context 1" in block
    assert "Context 2" in block
    assert "Article A" in block
    assert "Article B" in block


def test_build_prompt_includes_query_and_context():
    chunks = [_make_chunk("Style tip about blazers.")]
    prompt = build_prompt("How should I wear a blazer?", chunks)
    assert "blazer" in prompt.lower()
    assert "Style tip about blazers" in prompt
    assert "Cite sources" in prompt


def test_build_prompt_no_chunks():
    prompt = build_prompt("How should I style wide-leg pants?", [])
    assert "wide-leg" in prompt.lower()
    assert "No relevant context" in prompt
