"""Token-aware planning: decide how each file is fed to the LLM.

This module is where the token-limit best practices live. Each file becomes one
or more *work units*:

* A file that fits within the per-request input budget → a single unit.
* A file that exceeds the budget → split into several units using LlamaIndex's
  code-aware ``CodeSplitter`` (tree-sitter) when available, otherwise a
  token-based splitter. Per-chunk results are merged back in the extractor.

Splitting on syntactic boundaries (methods/classes) rather than blindly on
characters keeps each chunk semantically coherent, which materially improves
extraction quality on small models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Settings
from .repo_loader import SourceFile
from .token_utils import TokenCounter


@dataclass
class WorkUnit:
    """A single LLM-sized piece of work derived from a source file."""

    file: SourceFile
    text: str
    tokens: int
    chunk_index: int = 0
    chunk_count: int = 1

    @property
    def is_partial(self) -> bool:
        return self.chunk_count > 1


_LANG_BY_EXT = {
    ".java": "java",
    ".kt": "kotlin",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
}


def _reserve_for_prompt(budget: int) -> int:
    """Leave headroom in the input budget for the fixed prompt scaffolding."""
    # Prompt template + instructions cost a few hundred tokens; reserve ~15%.
    return max(512, int(budget * 0.85))


def _code_split(text: str, ext: str, max_chars: int) -> Optional[List[str]]:
    """Try LlamaIndex CodeSplitter; return chunks or None if unavailable."""
    lang = _LANG_BY_EXT.get(ext)
    if not lang:
        return None
    try:
        from llama_index.core.node_parser import CodeSplitter
        from llama_index.core.schema import Document

        splitter = CodeSplitter(
            language=lang,
            chunk_lines=60,
            chunk_lines_overlap=10,
            max_chars=max_chars,
        )
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        chunks = [n.get_content() for n in nodes if n.get_content().strip()]
        return chunks or None
    except Exception:
        return None


def _token_split(text: str, ext: str, chunk_token_budget: int, counter: TokenCounter) -> List[str]:
    """Token-based splitting fallback (LlamaIndex TokenTextSplitter or manual)."""
    try:
        from llama_index.core.node_parser import TokenTextSplitter
        from llama_index.core.schema import Document

        splitter = TokenTextSplitter(
            chunk_size=chunk_token_budget,
            chunk_overlap=int(chunk_token_budget * 0.1),
        )
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        chunks = [n.get_content() for n in nodes if n.get_content().strip()]
        if chunks:
            return chunks
    except Exception:
        pass

    # Last-resort manual splitter on line boundaries within the token budget.
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    buf: List[str] = []
    for line in lines:
        buf.append(line)
        if counter.count("".join(buf)) >= chunk_token_budget:
            chunks.append("".join(buf))
            buf = []
    if buf:
        chunks.append("".join(buf))
    return chunks or [text]


def plan_units(
    files: List[SourceFile],
    settings: Settings,
    counter: TokenCounter,
) -> List[WorkUnit]:
    """Convert files into token-budgeted work units."""
    input_budget = _reserve_for_prompt(settings.max_input_tokens_per_request)
    # rough chars-per-token to size the code splitter's max_chars
    max_chars = input_budget * 4

    units: List[WorkUnit] = []
    for f in files:
        if f.tokens <= input_budget:
            units.append(WorkUnit(file=f, text=f.content, tokens=f.tokens))
            continue

        ext = "." + f.rel_path.rsplit(".", 1)[-1] if "." in f.rel_path else ""
        chunks = _code_split(f.content, ext, max_chars) or _token_split(
            f.content, ext, input_budget, counter
        )
        # Ensure no chunk exceeds the hard budget (truncate as a guard).
        safe_chunks: List[str] = []
        for c in chunks:
            if counter.count(c) > settings.max_input_tokens_per_request:
                c = counter.truncate_to_budget(c, input_budget)
            safe_chunks.append(c)

        for i, c in enumerate(safe_chunks):
            units.append(
                WorkUnit(
                    file=f,
                    text=c,
                    tokens=counter.count(c),
                    chunk_index=i,
                    chunk_count=len(safe_chunks),
                )
            )
    return units


def summarize_plan(files: List[SourceFile], units: List[WorkUnit], settings: Settings) -> dict:
    """A compact, JSON-friendly description of the planned work (for --dry-run)."""
    total_tokens = sum(f.tokens for f in files)
    partials = [u for u in units if u.is_partial]
    split_files = {u.file.rel_path for u in partials}
    return {
        "files": len(files),
        "work_units": len(units),
        "files_requiring_split": len(split_files),
        "total_input_tokens_estimate": total_tokens,
        "max_input_tokens_per_request": settings.max_input_tokens_per_request,
        "largest_file": max(
            ((f.rel_path, f.tokens) for f in files),
            key=lambda x: x[1],
            default=("-", 0),
        ),
    }
