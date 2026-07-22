"""Prompt templates for the map (per-file) and reduce (synthesis) passes.

Prompts are deliberately compact and instruction-dense: they steer even small
models toward returning only the requested facts, which improves the odds of
valid structured output and keeps token usage down.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# MAP: analyse a single file / chunk
# --------------------------------------------------------------------------- #
MAP_SYSTEM = (
    "You are a senior software engineer performing static code analysis. "
    "You read one source file at a time and extract precise, factual structured "
    "information about it. You never invent methods or behaviour that is not in "
    "the code. You are concise. When unsure, prefer the most conservative answer."
)

MAP_HUMAN = """Analyze the following source file and extract structured knowledge.

Repository-relative path: {file_path}
Inferred component role (hint, verify against the code): {type_hint}
{chunk_note}

Guidelines:
- Identify the primary class/interface/enum and its package.
- Classify the component's architectural role.
- List the most important methods with their real signatures and a short
  description of what each does. For REST handlers, capture the HTTP verb and
  route path from the mapping annotations.
- Note key framework annotations, important dependencies (collaborators), and
  anything noteworthy (caching, transactions, security, N+1 risks, etc.).
- Rate the component's complexity as low, medium, or high.

SOURCE CODE:
```
{code}
```
"""

# --------------------------------------------------------------------------- #
# REDUCE (domain level): summarise one functional domain from component facts
# --------------------------------------------------------------------------- #
DOMAIN_SYSTEM = (
    "You are a software architect summarising one functional module of a system "
    "from per-file analysis facts. Be concise and accurate; do not invent details."
)

DOMAIN_HUMAN = """Summarise the '{domain}' domain of the project from the component
facts below. Capture its purpose, the most important components, a short summary
of any REST endpoints it exposes, and an overall complexity rating.

COMPONENT FACTS (JSON lines):
{facts}
"""

# --------------------------------------------------------------------------- #
# REDUCE (project level): synthesise the high-level overview
# --------------------------------------------------------------------------- #
OVERVIEW_SYSTEM = (
    "You are a principal engineer writing the executive overview of a codebase "
    "for other engineers. You synthesise from provided structured facts only. "
    "Be accurate, concrete, and concise."
)

OVERVIEW_HUMAN = """Produce a high-level overview of the project from the facts below.

Cover: the project's purpose and the problem it solves; a one-liner; the tech
stack; the architectural style and layers; the functional domains; the headline
features; and a short summary of the core domain/data model.

PROJECT SIGNALS:
- Repository: {repository}
- Detected build/config signals: {build_signals}
- Domains discovered: {domains}
- Component counts by type: {type_counts}

DOMAIN SUMMARIES (JSON lines):
{domain_summaries}
"""

# --------------------------------------------------------------------------- #
# REDUCE (project level): complexity + noteworthy synthesis
# --------------------------------------------------------------------------- #
COMPLEXITY_SYSTEM = (
    "You are a code quality analyst. From the provided per-component complexity "
    "facts, produce an overall complexity assessment and identify hotspots. "
    "Base everything on the facts; do not speculate beyond them."
)

COMPLEXITY_HUMAN = """Assess the overall complexity of the codebase and list the
main hotspots and cross-cutting observations.

PER-COMPONENT COMPLEXITY FACTS (JSON lines: path, type, complexity, method_count, noteworthy):
{facts}
"""

NOTEWORTHY_SYSTEM = (
    "You are a staff engineer doing a codebase review. From the aggregated notes "
    "below, distil the most important cross-cutting observations, grouping them by "
    "category (pattern, security, performance, testing, tech_debt, observation). "
    "Deduplicate and keep only genuinely noteworthy items."
)

NOTEWORTHY_HUMAN = """Aggregate and distil the noteworthy observations below into a
clean, deduplicated list of the most important items.

RAW NOTES (from per-file analysis):
{notes}
"""
