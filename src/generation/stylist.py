"""
LLM-powered fashion stylist that generates grounded outfit advice.

Takes retrieved context chunks and a user query, constructs a prompt,
and generates a response with source citations.
Supports OpenAI, HuggingFace Inference API, and Anthropic backends.
"""

import logging
import os
from dataclasses import dataclass

from src.pipeline.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fashion stylist assistant with a Gen-Z sensibility. Your job is to give \
practical, specific outfit and styling advice based on the fashion editorial content provided \
to you as context.

Rules:
1. Ground your advice in the provided context. Reference specific styling tips, trends, or \
recommendations from the articles.
2. When you use information from a source, cite it using its numbered reference, e.g. [1], [2]. Do not name sources inline.
3. If the context does not contain relevant information for the user's question, say so honestly \
rather than making things up. But still offer your best directional advice based on whatever \
context IS available, even if it is tangential.
4. Be specific: name colors, fabrics, silhouettes, and brands when the context supports it.
5. Keep your tone current and casual. Think TikTok fashion commentary, not a department store \
personal shopper. Reference aesthetic movements (quiet luxury, coquette, clean girl, dark \
academia, coastal cowgirl, etc.) when relevant. Avoid sounding like a millennial lifestyle blog.
6. Organize your response clearly. Use short paragraphs.
7. When discussing nostalgic or revival trends (Y2K, 90s, etc.), focus on how to style them in a \
modern way rather than recreating the original era literally."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    if not chunks:
        return "No relevant context found."

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        source_info = f"[{i}] {chunk.title}"
        if chunk.site:
            source_info += f" ({chunk.site})"
        if chunk.source_url:
            source_info += f"\nURL: {chunk.source_url}"

        blocks.append(f"--- [{i}] ---\n{source_info}\n\n{chunk.text}")

    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the full user prompt with retrieved context."""
    context = build_context_block(chunks)
    return (
        f"Here is relevant fashion editorial content to inform your response:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"User's question: {query}\n\n"
        f"Please provide styling advice grounded in the above context. "
        f"Cite sources using their reference numbers."
    )


@dataclass
class StylistResponse:
    """Response from the fashion stylist."""

    answer: str
    sources: list[dict]
    query: str


class FashionStylist:
    """Fashion styling assistant powered by RAG.

    Supports OpenAI, HuggingFace Inference API, and Anthropic for generation.
    """

    def __init__(self, provider: str | None = None):
        """Initialize the stylist with an LLM provider.

        Args:
            provider: "openai", "huggingface", or "anthropic". Auto-detected
                from env vars if not specified.
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self._client = None

    def _get_openai_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _generate_openai(self, prompt: str) -> str:
        """Generate a response using OpenAI's API."""
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def _generate_huggingface(self, prompt: str) -> str:
        """Generate a response using HuggingFace Inference API."""
        from huggingface_hub import InferenceClient

        model_id = os.getenv("HF_MODEL_ID", "HuggingFaceH4/zephyr-7b-beta")
        token = os.getenv("HF_API_TOKEN")

        client = InferenceClient(model=model_id, token=token)

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, prompt: str) -> str:
        """Generate a response using the Anthropic API."""
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> StylistResponse:
        """Generate a styled response for a user query.

        Args:
            query: The user's styling question.
            chunks: Retrieved context chunks from the vector store.

        Returns:
            StylistResponse with the answer, sources, and original query.
        """
        prompt = build_prompt(query, chunks)

        if self.provider == "openai":
            answer = self._generate_openai(prompt)
        elif self.provider == "huggingface":
            answer = self._generate_huggingface(prompt)
        elif self.provider == "anthropic":
            answer = self._generate_anthropic(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        # Collect unique sources
        seen_urls = set()
        sources = []
        for chunk in chunks:
            url = chunk.source_url
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": chunk.title,
                    "url": url,
                    "site": chunk.site,
                })

        return StylistResponse(answer=answer, sources=sources, query=query)
