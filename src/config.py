"""Runtime configuration, loaded from environment variables (and optional .env).

Every value has a default so the tool runs out of the box. Nothing here imports
an LLM library, so this module is safe to load in `--dry-run` mode with only the
core dependencies installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

try:  # optional convenience — load a local .env if python-dotenv is present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """All tunables for a run. Constructed from the environment by default."""

    # --- LLM selection (model-agnostic) ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5-20251001"
    synthesis_model: Optional[str] = None  # falls back to llm_model when empty
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048
    llm_api_base: Optional[str] = None  # for OpenAI-compatible local servers

    # --- Token budgeting ---
    max_input_tokens_per_request: int = 6000
    tokenizer_encoding: str = "cl100k_base"
    # Data-like files (SQL/CSV/JSON dumps) larger than this many tokens are
    # skipped — they are data, not code, and would waste the token budget.
    max_data_file_tokens: int = 20000

    # --- Structured output strategy ---
    # auto | native | parser
    structured_output_mode: str = "auto"

    # --- Scope / cost controls ---
    max_files: int = 0  # 0 = no limit
    max_concurrency: int = 4

    # --- File selection ---
    # Extensions considered "source" for analysis.
    source_extensions: tuple = (
        ".java", ".kt", ".sql", ".yaml", ".yml", ".properties", ".xml",
        ".gradle", ".kts", ".py", ".js", ".ts", ".go",
    )
    # Directory names to skip entirely.
    exclude_dirs: tuple = (
        ".git", "build", "target", "out", "node_modules", ".gradle",
        ".idea", "dist", "__pycache__", ".venv", "venv",
    )

    def resolved_synthesis_model(self) -> str:
        return self.synthesis_model or self.llm_model

    def public_dict(self) -> dict:
        """A JSON-safe view for embedding in the report metadata (no secrets)."""
        d = asdict(self)
        d["synthesis_model"] = self.resolved_synthesis_model()
        d.pop("source_extensions", None)
        d.pop("exclude_dirs", None)
        return d


def load_settings() -> Settings:
    """Build :class:`Settings` from environment variables."""
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
        synthesis_model=os.getenv("SYNTHESIS_MODEL") or None,
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.0),
        llm_max_output_tokens=_get_int("LLM_MAX_OUTPUT_TOKENS", 2048),
        llm_api_base=os.getenv("LLM_API_BASE") or None,
        max_input_tokens_per_request=_get_int("MAX_INPUT_TOKENS_PER_REQUEST", 6000),
        tokenizer_encoding=os.getenv("TOKENIZER_ENCODING", "cl100k_base"),
        max_data_file_tokens=_get_int("MAX_DATA_FILE_TOKENS", 20000),
        structured_output_mode=os.getenv("STRUCTURED_OUTPUT_MODE", "auto"),
        max_files=_get_int("MAX_FILES", 0),
        max_concurrency=_get_int("MAX_CONCURRENCY", 4),
    )
