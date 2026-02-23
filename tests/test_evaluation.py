"""Tests for the evaluation module."""

from src.evaluation.evaluate import evaluate_faithfulness, keyword_relevance_score
from src.pipeline.retriever import RetrievedChunk


def test_keyword_relevance_all_match():
    text = "A navy blazer with brown shoes looks great for spring."
    keywords = ["blazer", "navy", "brown", "shoes"]
    score = keyword_relevance_score(text, keywords)
    assert score == 1.0


def test_keyword_relevance_partial_match():
    text = "A navy blazer with white sneakers."
    keywords = ["blazer", "navy", "brown", "shoes"]
    score = keyword_relevance_score(text, keywords)
    assert score == 0.5  # 2 of 4 keywords match


def test_keyword_relevance_no_match():
    text = "Recipes for chocolate cake."
    keywords = ["blazer", "navy", "brown"]
    score = keyword_relevance_score(text, keywords)
    assert score == 0.0


def test_keyword_relevance_empty_keywords():
    score = keyword_relevance_score("some text", [])
    assert score == 0.0


def test_faithfulness_grounded_answer():
    chunks = [
        RetrievedChunk(
            text="Navy blazers pair exceptionally well with gray flannel trousers.",
            score=0.9,
            metadata={},
        )
    ]
    answer = "Navy blazers pair exceptionally well with gray flannel trousers for a classic look."
    score = evaluate_faithfulness(answer, chunks)
    assert score > 0.5


def test_faithfulness_ungrounded_answer():
    chunks = [
        RetrievedChunk(
            text="How to organize your closet by season.",
            score=0.5,
            metadata={},
        )
    ]
    answer = "Purple sneakers go great with neon yellow joggers and a cowboy hat."
    score = evaluate_faithfulness(answer, chunks)
    assert score == 0.0
