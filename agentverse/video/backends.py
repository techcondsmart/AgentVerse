"""Pluggable VLM caption backends — all free or free-tier, OpenAI-compatible.

A backend is just `(image_path, prompt) -> str`. The ensemble in
`captioning.py` calls two or more of these per frame and cross-checks them, so
the more *independent* the backends, the stronger the anti-hallucination
signal. Pair a hosted free-tier model with a local open-weights model for the
best independence/cost trade-off.

All hosted options below speak the OpenAI Chat Completions schema, so a single
`openai_vision_backend` factory covers them by swapping base_url + model + key.

Recommended free / free-tier backends (verify current limits before relying):
  - Google Gemini (OpenAI-compat endpoint), model e.g. "gemini-flash-latest"
      base_url = https://generativelanguage.googleapis.com/v1beta/openai/
      Generous free tier, strong vision + native video; great primary captioner.
  - OpenRouter free models (":free" suffix), e.g. a Qwen-VL or Llama-Vision
      base_url = https://openrouter.ai/api/v1
  - Groq (fast, free tier) for vision-capable models it hosts.
  - Local vLLM / Ollama serving Qwen2.5-VL / InternVL / MiniCPM-V — free
      forever, fully independent second opinion; base_url = your local server.

Keys come from env vars so nothing secret lives in code or config.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Optional

from agentverse.video.captioning import CaptionBackend


def _data_url(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def openai_vision_backend(
    model: str,
    base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> CaptionBackend:
    """Build a caption backend from any OpenAI-compatible vision endpoint.

    Works unchanged for OpenAI, Gemini (openai-compat), OpenRouter, Groq, and
    local vLLM/Ollama — you only vary `model`, `base_url`, and `api_key_env`.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get(api_key_env, "EMPTY"), base_url=base_url)

    def _backend(image_path: str, prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_path)},
                        },
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or ""

    return _backend


# Convenience presets for the recommended free/free-tier stack.
def gemini_backend(model: str = "gemini-flash-latest") -> CaptionBackend:
    return openai_vision_backend(
        model=model,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
    )


def openrouter_free_backend(model: str) -> CaptionBackend:
    """`model` should be a ':free' OpenRouter id, e.g. 'qwen/qwen2.5-vl-72b-instruct:free'."""
    return openai_vision_backend(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )


def local_vllm_backend(
    model: str, base_url: str = "http://localhost:8000/v1"
) -> CaptionBackend:
    """Local open-weights VLM (Qwen2.5-VL / InternVL / MiniCPM-V) via vLLM/Ollama."""
    return openai_vision_backend(model=model, base_url=base_url, api_key_env="VLLM_API_KEY")
