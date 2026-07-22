"""Orchestration: tie loading, chunking, map, and reduce into one pipeline.

Design choice: anything that can be computed deterministically from the code
(endpoint list, statistics, build signals) is computed in Python — we do not
spend LLM tokens on facts a parser can produce reliably. The LLM is reserved for
comprehension: descriptions, responsibilities, complexity judgements, overview.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, List, Optional

from . import __version__
from .chunker import WorkUnit, plan_units, summarize_plan
from .config import Settings
from .repo_loader import SourceFile, clone_repo, load_repository
from .schemas import (
    ApiEndpoint,
    CodebaseAnalysis,
    ComponentAnalysis,
    RunMetadata,
    Statistics,
)
from .token_utils import TokenCounter


# --------------------------------------------------------------------------- #
# Deterministic extraction (no LLM)
# --------------------------------------------------------------------------- #
def _detect_build_signals(files: List[SourceFile]) -> str:
    names = {os.path.basename(f.rel_path).lower() for f in files}
    signals = []
    if "build.gradle.kts" in names or "build.gradle" in names:
        signals.append("Gradle")
    if "pom.xml" in names:
        signals.append("Maven")
    if "package.json" in names:
        signals.append("npm/Node")
    if "go.mod" in names:
        signals.append("Go modules")
    if "requirements.txt" in names or "pyproject.toml" in names:
        signals.append("Python")
    if "application.yaml" in names or "application.yml" in names or "application.properties" in names:
        signals.append("Spring Boot config")
    return ", ".join(signals) if signals else "unknown"


def _endpoints_from_components(components: List[ComponentAnalysis]) -> List[ApiEndpoint]:
    endpoints: List[ApiEndpoint] = []
    for c in components:
        for m in c.methods:
            if m.http_method:
                endpoints.append(
                    ApiEndpoint(
                        http_method=m.http_method.upper(),
                        path=m.http_path or "",
                        handler=f"{c.class_name or c.file_path}.{m.name}",
                        description=m.description,
                        secured_role=(m.notes if m.notes and "ROLE" in (m.notes or "") else None),
                    )
                )
    endpoints.sort(key=lambda e: (e.path, e.http_method))
    return endpoints


def _statistics(
    files_discovered: int,
    components: List[ComponentAnalysis],
    endpoints: List[ApiEndpoint],
    loc: int,
    llm_calls: int,
) -> Statistics:
    by_type: dict = {}
    total_methods = 0
    for c in components:
        by_type[c.component_type.value] = by_type.get(c.component_type.value, 0) + 1
        total_methods += len(c.methods)
    return Statistics(
        files_discovered=files_discovered,
        files_analyzed=len(components),
        total_methods=total_methods,
        total_endpoints=len(endpoints),
        lines_of_code=loc,
        llm_calls=llm_calls,
        components_by_type=dict(sorted(by_type.items())),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def prepare(
    root: str,
    settings: Settings,
) -> tuple[List[SourceFile], List[WorkUnit], TokenCounter]:
    """Load + plan (no LLM). Shared by dry-run and full analysis."""
    counter = TokenCounter(settings.tokenizer_encoding)
    files = load_repository(root, settings, counter)
    if settings.max_files and settings.max_files > 0:
        files = files[: settings.max_files]
    units = plan_units(files, settings, counter)
    return files, units, counter


def dry_run(root: str, settings: Settings) -> dict:
    """Run everything except LLM calls; report the plan and token budget."""
    files, units, _ = prepare(root, settings)
    plan = summarize_plan(files, units, settings)
    loader = getattr(load_repository, "last_loader", "builtin")
    return {
        "loader": loader,
        "build_signals": _detect_build_signals(files),
        "components_by_type_hint": _type_hint_counts(files),
        "domains": sorted({f.domain for f in files}),
        "plan": plan,
    }


def _type_hint_counts(files: List[SourceFile]) -> dict:
    counts: dict = {}
    for f in files:
        counts[f.type_hint] = counts.get(f.type_hint, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def analyze(
    root: str,
    settings: Settings,
    repository_name: str,
    map_progress: Optional[Callable] = None,
    log: Optional[Callable] = None,
) -> CodebaseAnalysis:
    """Full pipeline: load → map → reduce → assemble document."""
    from .extractor import run_map
    from .llm_factory import build_map_llm, build_synthesis_llm, describe
    from .synthesizer import (
        synthesize_complexity,
        synthesize_domains,
        synthesize_noteworthy,
        synthesize_overview,
    )

    def _log(msg: str):
        if log:
            log(msg)

    files, units, counter = prepare(root, settings)
    total_loc = sum(f.lines for f in files)
    _log(f"Loaded {len(files)} files -> {len(units)} work units "
         f"(loader={getattr(load_repository, 'last_loader', 'builtin')})")

    # ---- map ---------------------------------------------------------------
    map_llm = build_map_llm(settings)
    _log("Running map pass (per-file extraction)...")
    components, map_stats = run_map(map_llm, units, settings, progress=map_progress)
    _log(f"Map complete: {len(components)} components, {map_stats.llm_calls} calls, "
         f"{map_stats.failures} failures, modes={map_stats.modes}")

    # ---- reduce ------------------------------------------------------------
    synth_llm = build_synthesis_llm(settings)
    _log("Synthesising domain summaries...")
    domain_summaries = synthesize_domains(synth_llm, components, settings, counter)
    build_signals = _detect_build_signals(files)
    type_counts = {}
    for c in components:
        type_counts[c.component_type.value] = type_counts.get(c.component_type.value, 0) + 1

    _log("Synthesising project overview...")
    overview = synthesize_overview(
        synth_llm, domain_summaries, build_signals, repository_name, type_counts, settings, counter
    )
    _log("Synthesising complexity analysis...")
    complexity = synthesize_complexity(synth_llm, components, settings, counter)
    _log("Synthesising noteworthy observations...")
    noteworthy = synthesize_noteworthy(synth_llm, components, settings, counter)

    # ---- deterministic assembly -------------------------------------------
    endpoints = _endpoints_from_components(components)
    stats = _statistics(len(files), components, endpoints, total_loc, map_stats.llm_calls)

    metadata = RunMetadata(
        repository=repository_name,
        analyzer_version=__version__,
        generated_at=datetime.now(timezone.utc).isoformat(),
        llm=describe(settings),
        token_budget={
            "max_input_tokens_per_request": settings.max_input_tokens_per_request,
            "tokenizer_encoding": settings.tokenizer_encoding,
        },
        notes=[
            f"structured-output modes used: {map_stats.modes}",
            f"map failures: {map_stats.failures}",
        ],
    )

    return CodebaseAnalysis(
        metadata=metadata,
        statistics=stats,
        project_overview=overview,
        domains=domain_summaries,
        api_endpoints=endpoints,
        complexity_analysis=complexity,
        noteworthy=noteworthy.items,
        components=components,
    )
