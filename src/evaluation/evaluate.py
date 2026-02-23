"""
Evaluation metrics for the fashion stylist RAG pipeline.

Measures retrieval quality (precision, recall) and answer quality
(faithfulness, relevance) using a small labeled test set.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.pipeline.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class EvalResult:
    """Evaluation results for a single query."""

    query: str
    precision: float
    recall: float
    retrieved_urls: list[str]
    relevant_urls: list[str]
    hit_urls: list[str]


@dataclass
class EvalSummary:
    """Aggregate evaluation metrics across all test queries."""

    mean_precision: float
    mean_recall: float
    mean_mrr: float
    num_queries: int
    results: list[EvalResult]

    def to_dict(self) -> dict:
        return {
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "num_queries": self.num_queries,
        }


# Sample test queries with expected relevant article URLs.
# In practice, populate this after scraping by manually labeling
# which articles are relevant to each query.
SAMPLE_TEST_SET = [
    {
        "query": "How should I style a navy blazer with brown shoes?",
        "relevant_keywords": ["blazer", "navy", "brown", "shoes", "dress shoes"],
    },
    {
        "query": "What pants go well with Chelsea boots?",
        "relevant_keywords": ["chelsea", "boots", "pants", "trousers", "jeans"],
    },
    {
        "query": "How do I dress for a casual date night?",
        "relevant_keywords": ["date", "casual", "evening", "dinner", "night out"],
    },
    {
        "query": "What are the best layering pieces for spring?",
        "relevant_keywords": ["spring", "layer", "jacket", "cardigan", "lightweight"],
    },
    {
        "query": "How to wear wide-leg trousers?",
        "relevant_keywords": ["wide-leg", "trousers", "pants", "silhouette", "relaxed"],
    },
    {
        "query": "What goes with a black leather jacket?",
        "relevant_keywords": ["leather jacket", "black", "edgy", "moto", "casual"],
    },
    {
        "query": "Best sneakers for smart casual outfits?",
        "relevant_keywords": ["sneakers", "smart casual", "clean", "minimalist", "white"],
    },
    {
        "query": "How to dress up a t-shirt?",
        "relevant_keywords": ["t-shirt", "tee", "elevated", "blazer", "layer"],
    },
]


def keyword_relevance_score(text: str, keywords: list[str]) -> float:
    """Score how relevant a chunk is based on keyword matches."""
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords) if keywords else 0.0


def evaluate_retrieval(
    retriever: Retriever,
    test_set: list[dict] | None = None,
    top_k: int = 5,
    relevance_threshold: float = 0.3,
) -> EvalSummary:
    """Evaluate retrieval quality on a test set.

    Uses keyword-based relevance scoring when manually labeled
    relevant URLs are not available.

    Args:
        retriever: The Retriever instance to evaluate.
        test_set: List of test queries with relevant_keywords or relevant_urls.
        top_k: Number of chunks to retrieve per query.
        relevance_threshold: Minimum keyword score to consider a chunk relevant.

    Returns:
        EvalSummary with aggregate metrics.
    """
    test_set = test_set or SAMPLE_TEST_SET
    results = []
    mrr_scores = []

    for test_case in test_set:
        query = test_case["query"]
        keywords = test_case.get("relevant_keywords", [])

        chunks = retriever.retrieve(query, top_k=top_k)

        if not chunks:
            results.append(EvalResult(
                query=query,
                precision=0.0,
                recall=0.0,
                retrieved_urls=[],
                relevant_urls=[],
                hit_urls=[],
            ))
            mrr_scores.append(0.0)
            continue

        # Score each chunk by keyword relevance
        relevance_scores = [
            keyword_relevance_score(chunk.text, keywords)
            for chunk in chunks
        ]

        relevant_count = sum(1 for s in relevance_scores if s >= relevance_threshold)
        precision = relevant_count / len(chunks) if chunks else 0.0

        # For recall, we estimate based on proportion of keywords hit
        all_text = " ".join(c.text for c in chunks)
        keywords_found = sum(1 for kw in keywords if kw.lower() in all_text.lower())
        recall = keywords_found / len(keywords) if keywords else 0.0

        # MRR: rank of the first relevant result
        first_relevant_rank = 0
        for i, score in enumerate(relevance_scores):
            if score >= relevance_threshold:
                first_relevant_rank = i + 1
                break
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
        mrr_scores.append(mrr)

        retrieved_urls = [c.source_url for c in chunks]
        results.append(EvalResult(
            query=query,
            precision=precision,
            recall=recall,
            retrieved_urls=retrieved_urls,
            relevant_urls=[],
            hit_urls=[],
        ))

    summary = EvalSummary(
        mean_precision=float(np.mean([r.precision for r in results])),
        mean_recall=float(np.mean([r.recall for r in results])),
        mean_mrr=float(np.mean(mrr_scores)),
        num_queries=len(results),
        results=results,
    )

    logger.info(
        f"Evaluation complete: P={summary.mean_precision:.3f}, "
        f"R={summary.mean_recall:.3f}, MRR={summary.mean_mrr:.3f} "
        f"over {summary.num_queries} queries"
    )
    return summary


def save_eval_results(summary: EvalSummary, filename: str = "eval_results.json"):
    """Save evaluation results to a JSON file."""
    output_path = EVAL_DIR / filename
    with open(output_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    logger.info(f"Saved evaluation results to {output_path}")


def evaluate_faithfulness(answer: str, chunks: list[RetrievedChunk]) -> float:
    """Estimate answer faithfulness by checking if claims in the answer
    are grounded in the retrieved context.

    Simple heuristic: measures what fraction of sentences in the answer
    share significant n-gram overlap with the context.
    """
    context_text = " ".join(c.text.lower() for c in chunks)

    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    grounded_count = 0
    for sentence in sentences:
        words = sentence.lower().split()
        # Check for 3-gram overlap with context
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        if any(tg in context_text for tg in trigrams):
            grounded_count += 1

    return grounded_count / len(sentences)


if __name__ == "__main__":
    retriever = Retriever()
    summary = evaluate_retrieval(retriever)
    print("\nRetrieval Evaluation Results:")
    print(f"  Mean Precision: {summary.mean_precision:.3f}")
    print(f"  Mean Recall:    {summary.mean_recall:.3f}")
    print(f"  Mean MRR:       {summary.mean_mrr:.3f}")
    print(f"  Queries tested: {summary.num_queries}")
    save_eval_results(summary)
