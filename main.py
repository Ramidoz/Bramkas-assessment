#!/usr/bin/env python3
"""Codebase Analyzer — CLI entrypoint.

Examples
--------
Dry run (no API key needed) — verify discovery, chunking, and token budget:

    python main.py --repo-url https://github.com/codejsha/spring-rest-sakila --dry-run

Full analysis against a local checkout, writing JSON:

    LLM_PROVIDER=anthropic LLM_MODEL=claude-haiku-4-5-20251001 \\
    python main.py --repo-path ./spring-rest-sakila --output output/sakila_analysis.json

Use a local, free model via an OpenAI-compatible server (Ollama):

    LLM_PROVIDER=openai LLM_MODEL=qwen2.5-coder:7b \\
    LLM_API_BASE=http://localhost:11434/v1 \\
    python main.py --repo-path ./spring-rest-sakila
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from src.analyzer import analyze, dry_run
from src.config import load_settings
from src.repo_loader import clone_repo

DEFAULT_REPO_URL = "https://github.com/codejsha/spring-rest-sakila"


def _resolve_repo(args) -> tuple[str, str]:
    """Return (root_path, repository_name)."""
    if args.repo_path:
        root = os.path.abspath(args.repo_path)
        if not os.path.isdir(root):
            print(f"error: --repo-path '{root}' is not a directory", file=sys.stderr)
            sys.exit(2)
        return root, args.repo_name or os.path.basename(root.rstrip("/"))

    url = args.repo_url or DEFAULT_REPO_URL
    dest = args.clone_dir or os.path.join(tempfile.gettempdir(), "codebase-analyzer-src")
    print(f"Cloning {url} -> {dest} ...")
    root = clone_repo(url, dest)
    name = args.repo_name or url.rstrip("/").split("/")[-1].replace(".git", "")
    return root, name


def _progress(done: int, total: int, path: str) -> None:
    pct = int(done * 100 / max(1, total))
    sys.stderr.write(f"\r  map: {done}/{total} ({pct}%)  {path[:60]:<60}")
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-powered codebase knowledge extractor.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--repo-url", help="Git URL to clone and analyze.")
    src.add_argument("--repo-path", help="Path to an existing local checkout.")
    parser.add_argument("--repo-name", help="Override the repository display name.")
    parser.add_argument("--clone-dir", help="Where to clone --repo-url (default: temp dir).")
    parser.add_argument(
        "--output", "-o", default="output/analysis.json", help="Output JSON path."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only: discover files, chunk, and report token budget without calling the LLM.",
    )
    parser.add_argument("--max-files", type=int, help="Override MAX_FILES.")
    args = parser.parse_args()

    settings = load_settings()
    if args.max_files is not None:
        settings.max_files = args.max_files

    root, repo_name = _resolve_repo(args)
    print(f"Analyzing '{repo_name}' at {root}")
    print(
        f"LLM: provider={settings.llm_provider} model={settings.llm_model} "
        f"synthesis={settings.resolved_synthesis_model()} "
        f"budget={settings.max_input_tokens_per_request} tok/req\n"
    )

    if args.dry_run:
        report = dry_run(root, settings)
        print(json.dumps(report, indent=2))
        print("\nDry run complete — no LLM calls were made.")
        return 0

    result = analyze(
        root,
        settings,
        repository_name=repo_name,
        map_progress=_progress,
        log=lambda m: print(m),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result.model_dump(mode="json"), fh, indent=2, ensure_ascii=False)

    s = result.statistics
    print(
        f"\nDone. Wrote {args.output}\n"
        f"  files analyzed : {s.files_analyzed}/{s.files_discovered}\n"
        f"  methods        : {s.total_methods}\n"
        f"  endpoints      : {s.total_endpoints}\n"
        f"  LLM calls      : {s.llm_calls}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
