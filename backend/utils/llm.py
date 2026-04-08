"""
NVIDIA NIM LLM client — used for query rewriting, HyDE, multi-query, and agentic RAG.

Uses the OpenAI-compatible API that NVIDIA NIM exposes.
Not needed until Technique 4 (Query Rewriting).
"""

from openai import OpenAI
from config.settings import settings

# NVIDIA NIM uses an OpenAI-compatible endpoint
_client = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.nvidia_nim_api_key,
        )
    return _client


def llm_complete(prompt: str, model: str = "meta/llama-3.1-70b-instruct") -> str:
    """Send a prompt to NVIDIA NIM and return the response text."""
    client = get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content
