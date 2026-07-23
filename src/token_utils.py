"""Provider-neutral token counting and budgeting.

Token limits are the central constraint of this project. We count tokens with
`tiktoken` (offline, no API calls) using a fixed encoding as a good cross-model
approximation. If tiktoken is unavailable we fall back to a conservative
characters/4 heuristic so the tool still runs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _encoder(encoding_name: str) -> Callable[[str], int]:
    """Return a function that counts tokens for `text` using the given encoding."""
    try:
        import tiktoken

        try:
            enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            enc = tiktoken.get_encoding(_DEFAULT_ENCODING)

        def _count(text: str) -> int:
            return len(enc.encode(text, disallowed_special=()))

        return _count
    except Exception:
        # Heuristic fallback: ~4 characters per token.
        def _count(text: str) -> int:
            return max(1, len(text) // 4)

        return _count


class TokenCounter:
    """Small stateful wrapper around a chosen encoding."""

    def __init__(self, encoding_name: str = _DEFAULT_ENCODING) -> None:
        self.encoding_name = encoding_name
        self._count = _encoder(encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return self._count(text)

    def fits(self, text: str, budget: int) -> bool:
        return self.count(text) <= budget

    def truncate_to_budget(self, text: str, budget: int) -> str:
        """Return a prefix of `text` that fits within `budget` tokens.

        Uses a proportional estimate then trims until it fits — cheap and good
        enough for guardrails (real splitting is done by the chunker).
        """
        if self.fits(text, budget):
            return text
        approx_chars = max(1, int(len(text) * budget / max(1, self.count(text))))
        candidate = text[:approx_chars]
        while candidate and self.count(candidate) > budget:
            candidate = candidate[: int(len(candidate) * 0.9)]
        return candidate
