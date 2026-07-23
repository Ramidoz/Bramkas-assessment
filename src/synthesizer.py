"""Reduce pass — synthesise whole-project knowledge from per-file facts.

This is a hierarchical map-reduce to respect token limits:

    components ──group by domain──▶ DomainSummary (per domain)
    DomainSummaries ─────────────▶ ProjectOverview
    component complexity facts ──▶ ComplexityAnalysis
    aggregated notes ────────────▶ Noteworthy list

The reduce pass never re-sends source code — only the compact structured facts
produced by the map pass — so it stays well within the input budget even for
large codebases.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional

from .config import Settings
from .prompts import (
    COMPLEXITY_HUMAN,
    COMPLEXITY_SYSTEM,
    DOMAIN_HUMAN,
    DOMAIN_SYSTEM,
    NOTEWORTHY_HUMAN,
    NOTEWORTHY_SYSTEM,
    OVERVIEW_HUMAN,
    OVERVIEW_SYSTEM,
)
from .schemas import (
    ComplexityAnalysis,
    ComponentAnalysis,
    DomainSummary,
    NoteworthyList,
    ProjectOverview,
)
from .structured import extract_structured
from .token_utils import TokenCounter


def _compact_component(c: ComponentAnalysis) -> dict:
    """A small, token-cheap fact record for a component (no full method bodies)."""
    return {
        "path": c.file_path,
        "class": c.class_name,
        "type": c.component_type.value,
        "responsibility": c.responsibility,
        "methods": [m.name for m in c.methods],
        "complexity": c.complexity.value,
    }


def _fit_lines(lines: List[str], budget: int, counter: TokenCounter) -> str:
    """Join as many `lines` as fit within `budget` tokens (best effort)."""
    out: List[str] = []
    used = 0
    for line in lines:
        cost = counter.count(line) + 1
        if used + cost > budget:
            out.append(f"... ({len(lines) - len(out)} more omitted for token budget)")
            break
        out.append(line)
        used += cost
    return "\n".join(out)


def synthesize_domains(
    llm,
    components: List[ComponentAnalysis],
    settings: Settings,
    counter: TokenCounter,
) -> List[DomainSummary]:
    by_domain: Dict[str, List[ComponentAnalysis]] = defaultdict(list)
    for c in components:
        domain = getattr(c, "_domain", None) or _domain_of(c)
        by_domain[domain].append(c)

    budget = int(settings.max_input_tokens_per_request * 0.7)
    summaries: List[DomainSummary] = []
    for domain, comps in sorted(by_domain.items()):
        facts = _fit_lines(
            [json.dumps(_compact_component(c)) for c in comps], budget, counter
        )
        human = DOMAIN_HUMAN.format(domain=domain, facts=facts)
        res = extract_structured(
            llm, DomainSummary, DOMAIN_SYSTEM, human, mode=settings.structured_output_mode
        )
        if res.value is not None:
            res.value.domain = domain
            summaries.append(res.value)
        else:
            summaries.append(
                DomainSummary(
                    domain=domain,
                    purpose="(synthesis unavailable)",
                    key_components=[c.class_name or c.file_path for c in comps[:5]],
                )
            )
    return summaries


def _domain_of(c: ComponentAnalysis) -> str:
    parts = c.file_path.replace("\\", "/").split("/")
    for marker in ("services", "modules", "features", "domains"):
        if marker in parts:
            i = parts.index(marker)
            if i + 1 < len(parts) - 1:
                return parts[i + 1]
    for marker in ("config", "common", "security"):
        if marker in parts:
            return marker
    return parts[-2] if len(parts) >= 2 else "root"


def synthesize_overview(
    llm,
    domain_summaries: List[DomainSummary],
    build_signals: str,
    repository: str,
    type_counts: dict,
    settings: Settings,
    counter: TokenCounter,
) -> Optional[ProjectOverview]:
    budget = int(settings.max_input_tokens_per_request * 0.7)
    summaries_text = _fit_lines(
        [s.model_dump_json() for s in domain_summaries], budget, counter
    )
    human = OVERVIEW_HUMAN.format(
        repository=repository,
        build_signals=build_signals,
        domains=", ".join(sorted({s.domain for s in domain_summaries})),
        type_counts=json.dumps(type_counts),
        domain_summaries=summaries_text,
    )
    res = extract_structured(
        llm, ProjectOverview, OVERVIEW_SYSTEM, human, mode=settings.structured_output_mode
    )
    return res.value


def synthesize_complexity(
    llm,
    components: List[ComponentAnalysis],
    settings: Settings,
    counter: TokenCounter,
) -> Optional[ComplexityAnalysis]:
    budget = int(settings.max_input_tokens_per_request * 0.7)
    facts = [
        json.dumps(
            {
                "path": c.file_path,
                "type": c.component_type.value,
                "complexity": c.complexity.value,
                "method_count": len(c.methods),
                "noteworthy": c.noteworthy[:2],
            }
        )
        for c in components
    ]
    human = COMPLEXITY_HUMAN.format(facts=_fit_lines(facts, budget, counter))
    res = extract_structured(
        llm, ComplexityAnalysis, COMPLEXITY_SYSTEM, human, mode=settings.structured_output_mode
    )
    return res.value


def synthesize_noteworthy(
    llm,
    components: List[ComponentAnalysis],
    settings: Settings,
    counter: TokenCounter,
) -> NoteworthyList:
    budget = int(settings.max_input_tokens_per_request * 0.7)
    notes: List[str] = []
    for c in components:
        for n in c.noteworthy:
            notes.append(f"[{c.component_type.value}:{c.file_path}] {n}")
    if not notes:
        return NoteworthyList(items=[])
    human = NOTEWORTHY_HUMAN.format(notes=_fit_lines(notes, budget, counter))
    res = extract_structured(
        llm, NoteworthyList, NOTEWORTHY_SYSTEM, human, mode=settings.structured_output_mode
    )
    return res.value or NoteworthyList(items=[])
