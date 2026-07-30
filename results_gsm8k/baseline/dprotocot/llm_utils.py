# -*- coding: utf-8 -*-
"""
Lightweight LLM-call helpers for baseline methods (USC, GenSelect, etc.).

Supports:
  - vLLM (OpenAI-compatible server, the default for local GPUs)
  - OpenAI / Together AI / any OpenAI-compatible endpoint
  - A dry-run mock for testing

Usage::

    llm = get_llm(backend="vllm", base_url="http://localhost:8000/v1", model="qwen3-8b")
    response = llm([{"role": "user", "content": "Hello"}])

The returned callable matches the ``LLMCall`` protocol used in baselines.py.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

LLMCall = Callable[[List[Dict[str, str]]], str]


def _openai_compatible(
    base_url: str,
    model: str,
    api_key: str = "not-needed",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> LLMCall:
    """Return an llm_call for any OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _call(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _call


def _dry_run(messages: List[Dict[str, str]]) -> str:
    """Mock LLM for testing -- always returns Path 1."""
    return "Path 1"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def vllm(
    base_url: str = "http://localhost:8000/v1",
    model: str = "qwen3-8b",
    **kw,
) -> LLMCall:
    """vLLM OpenAI-compatible server (default for local GPU)."""
    return _openai_compatible(base_url=base_url, model=model, **kw)


def together(
    model: str = "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    api_key: Optional[str] = None,
    **kw,
) -> LLMCall:
    """Together AI endpoint."""
    key = api_key or os.environ.get("TOGETHER_API_KEY", "")
    return _openai_compatible(
        base_url="https://api.together.xyz/v1",
        model=model,
        api_key=key,
        **kw,
    )


def deepinfra(
    model: str = "meta-llama/Llama-3.1-70B-Instruct",
    api_key: Optional[str] = None,
    **kw,
) -> LLMCall:
    """DeepInfra endpoint (fast + cheap)."""
    key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
    return _openai_compatible(
        base_url="https://api.deepinfra.com/v1/openai",
        model=model,
        api_key=key,
        **kw,
    )


def openai(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    **kw,
) -> LLMCall:
    """Standard OpenAI endpoint."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    return _openai_compatible(
        base_url="https://api.openai.com/v1",
        model=model,
        api_key=key,
        **kw,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
BACKENDS = {
    "vllm": vllm,
    "together": together,
    "deepinfra": deepinfra,
    "openai": openai,
    "dry-run": lambda **kw: _dry_run,
}


def get_llm(backend: str = "vllm", **kw) -> LLMCall:
    """Factory: return an llm_call function for the given backend.

    Examples::

        llm = get_llm("vllm", base_url="http://localhost:8000/v1", model="qwen3-8b")
        llm = get_llm("together", model="meta-llama/Llama-3.1-70B-Instruct-Turbo")
        llm = get_llm("dry-run")   # always returns "Path 1"
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.  Choose from {list(BACKENDS)}.")
    return BACKENDS[backend](**kw)
