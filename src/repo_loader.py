"""Repository ingestion: clone (optional), discover, and load source files.

Loading is done with **LlamaIndex** (``SimpleDirectoryReader``) so we benefit
from its document abstraction and metadata handling. We control file *selection*
ourselves (extension + excluded-directory filtering, importance ranking) and
hand the chosen paths to LlamaIndex to read. If LlamaIndex is unavailable, a
plain-``open`` fallback keeps the tool runnable.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Settings
from .token_utils import TokenCounter


# Ranking of component types by architectural importance (used when MAX_FILES
# forces us to pick a subset — analyse the most informative files first).
_IMPORTANCE = {
    "controller": 100,
    "service": 90,
    "repository": 80,
    "entity": 70,
    "config": 65,
    "security": 60,
    "mapper": 55,
    "assembler": 50,
    "dto": 45,
    "converter": 40,
    "filter": 38,
    "exception": 35,
    "application": 34,
    "serializer": 30,
    "constant": 25,
    "util": 20,
    "other": 10,
}


@dataclass
class SourceFile:
    path: str  # absolute path on disk
    rel_path: str  # repository-relative path
    content: str
    type_hint: str
    domain: str
    lines: int
    tokens: int = 0
    importance: int = field(default=10)


def _infer_type(rel_path: str, filename: str) -> str:
    name = filename.lower()
    p = rel_path.lower()
    # filename-suffix signals first (most specific)
    suffix_map = [
        ("controller", "controller"),
        ("serviceimpl", "service"),
        ("service", "service"),
        ("repositoryimpl", "repository"),
        ("repository", "repository"),
        ("entity", "entity"),
        ("assembler", "assembler"),
        ("converter", "converter"),
        ("serializer", "serializer"),
        ("mapper", "mapper"),
        ("dto", "dto"),
        ("config", "config"),
        ("application", "application"),
        ("filter", "filter"),
        ("exception", "exception"),
    ]
    stem = name.rsplit(".", 1)[0]
    for needle, t in suffix_map:
        if stem.endswith(needle):
            return t
    # directory signals
    dir_map = [
        ("/controller", "controller"),
        ("/service", "service"),
        ("/repository", "repository"),
        ("/entity", "entity"),
        ("/dto", "dto"),
        ("/mapper", "mapper"),
        ("/assembler", "assembler"),
        ("/converter", "converter"),
        ("/serializer", "serializer"),
        ("/security", "security"),
        ("/filter", "filter"),
        ("/exception", "exception"),
        ("/constant", "constant"),
        ("/config", "config"),
        ("/util", "util"),
    ]
    for needle, t in dir_map:
        if needle in p:
            return t
    return "other"


def _infer_domain(rel_path: str) -> str:
    """Best-effort functional-domain extraction from a path.

    For layouts like ``.../services/<domain>/...`` returns ``<domain>``.
    Falls back to a sensible parent directory name otherwise.
    """
    parts = rel_path.replace("\\", "/").split("/")
    for marker in ("services", "modules", "features", "domains"):
        if marker in parts:
            i = parts.index(marker)
            if i + 1 < len(parts) - 1:  # not the file itself
                return parts[i + 1]
    # non-service files: group by a meaningful directory
    for marker in ("config", "common", "security"):
        if marker in parts:
            return marker
    # otherwise the directory immediately containing the file
    return parts[-2] if len(parts) >= 2 else "root"


def clone_repo(repo_url: str, dest_dir: str) -> str:
    """Shallow-clone `repo_url` into `dest_dir` (idempotent). Returns the path."""
    if os.path.isdir(os.path.join(dest_dir, ".git")):
        return dest_dir
    os.makedirs(os.path.dirname(dest_dir) or ".", exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, dest_dir],
        check=True,
        capture_output=True,
        text=True,
    )
    return dest_dir


def _discover_paths(root: str, settings: Settings) -> List[str]:
    exts = tuple(settings.source_extensions)
    excluded = set(settings.exclude_dirs)
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excluded]
        for fn in filenames:
            if fn.endswith(exts):
                found.append(os.path.join(dirpath, fn))
    return found


def _read_with_llamaindex(paths: List[str]) -> Optional[dict]:
    """Return {path: text} using LlamaIndex, or None if it isn't available."""
    try:
        from llama_index.core import SimpleDirectoryReader
    except Exception:
        return None
    try:
        reader = SimpleDirectoryReader(input_files=paths, errors="ignore")
        docs = reader.load_data()
    except Exception:
        return None
    out: dict = {}
    for d in docs:
        fp = d.metadata.get("file_path") or d.metadata.get("filename")
        if fp is None:
            continue
        out.setdefault(os.path.abspath(fp), "")
        out[os.path.abspath(fp)] += d.text
    return out


def load_repository(
    root: str,
    settings: Settings,
    counter: TokenCounter,
) -> List[SourceFile]:
    """Discover and load source files under `root` into :class:`SourceFile`s."""
    root = os.path.abspath(root)
    paths = _discover_paths(root, settings)

    contents = _read_with_llamaindex(paths)
    loaded_via = "llamaindex" if contents is not None else "builtin"

    files: List[SourceFile] = []
    for p in paths:
        abs_p = os.path.abspath(p)
        text: Optional[str]
        if contents is not None:
            text = contents.get(abs_p)
            if text is None:  # LlamaIndex skipped it; read directly
                text = _safe_read(abs_p)
        else:
            text = _safe_read(abs_p)
        if not text:
            continue
        rel = os.path.relpath(abs_p, root)
        filename = os.path.basename(abs_p)
        n_tokens = counter.count(text)

        # Skip oversized data-like files (SQL/CSV/JSON dumps): they are data,
        # not code, and would consume the token budget without adding insight.
        ext = os.path.splitext(filename)[1].lower()
        if ext in {".sql", ".csv", ".json", ".txt"} and n_tokens > settings.max_data_file_tokens:
            continue

        type_hint = _infer_type(rel, filename)
        files.append(
            SourceFile(
                path=abs_p,
                rel_path=rel,
                content=text,
                type_hint=type_hint,
                domain=_infer_domain(rel),
                lines=text.count("\n") + 1,
                tokens=n_tokens,
                importance=_IMPORTANCE.get(type_hint, 10),
            )
        )

    # Deterministic, importance-first ordering (then by path for stability).
    files.sort(key=lambda f: (-f.importance, f.rel_path))
    setattr(load_repository, "last_loader", loaded_via)
    return files


def _safe_read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception:
        return None
