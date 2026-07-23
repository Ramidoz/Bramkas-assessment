"""Model-agnostic LLM construction, built on LangChain's ``init_chat_model``.

The whole pipeline talks to *a* chat model through this factory and never to a
specific vendor SDK. Swapping providers (Anthropic ↔ OpenAI ↔ a local Ollama /
LM Studio / vLLM server) is a matter of environment variables — no code change.

``init_chat_model`` supports many providers out of the box (anthropic, openai,
google_genai, groq, mistralai, ollama, bedrock, fireworks, together, ...). We
add a thin OpenAI-compatible path (``base_url``) so any self-hosted, cheap, or
free local model that speaks the OpenAI API also works.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import Settings


class LLMUnavailableError(RuntimeError):
    """Raised when a chat model cannot be constructed (missing dep / key)."""


def _build(
    provider: str,
    model: str,
    settings: Settings,
    **overrides: Any,
):
    """Construct a LangChain chat model for `provider`/`model`.

    Kept lazy: the provider integration package is only imported when actually
    requested, so users install just the one provider they need.
    """
    from langchain.chat_models import init_chat_model

    kwargs: dict[str, Any] = {
        "model": model,
        "model_provider": provider,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_output_tokens,
    }

    # OpenAI-compatible local/self-hosted servers: route through the openai
    # integration but point it at a custom base URL (Ollama, LM Studio, vLLM…).
    if settings.llm_api_base and provider in {"openai", "azure_openai"}:
        kwargs["base_url"] = settings.llm_api_base
        # Local servers usually accept any key; provide a harmless default.
        kwargs.setdefault("api_key", "not-needed")

    kwargs.update(overrides)

    try:
        return init_chat_model(**kwargs)
    except ImportError as exc:  # provider package not installed
        raise LLMUnavailableError(
            f"Provider '{provider}' is not installed. "
            f"Install its LangChain integration (e.g. `pip install langchain-{provider}`). "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:  # bad model id, missing key surfaced at call time, etc.
        raise LLMUnavailableError(
            f"Could not initialise model '{model}' via provider '{provider}': {exc}"
        ) from exc


def build_map_llm(settings: Settings, **overrides: Any):
    """The (cheap) model used for the per-file map pass."""
    return _build(settings.llm_provider, settings.llm_model, settings, **overrides)


def build_synthesis_llm(settings: Settings, **overrides: Any):
    """The model used for the reduce/synthesis pass (may equal the map model)."""
    return _build(
        settings.llm_provider,
        settings.resolved_synthesis_model(),
        settings,
        **overrides,
    )


def describe(settings: Settings) -> dict:
    """Human/JSON-friendly description of the active model configuration."""
    return {
        "provider": settings.llm_provider,
        "map_model": settings.llm_model,
        "synthesis_model": settings.resolved_synthesis_model(),
        "temperature": settings.llm_temperature,
        "api_base": settings.llm_api_base,
        "structured_output_mode": settings.structured_output_mode,
    }
