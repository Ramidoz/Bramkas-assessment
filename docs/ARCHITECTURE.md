# Architecture & Design Walkthrough

This document explains how the analyzer works, module by module, and the reasoning behind
the key decisions. It complements the top-level `README.md`.

## 1. Problem framing

We must feed a ~10k-LOC, ~188-file Java project to an LLM and get **structured, machine-readable
knowledge** back — without exceeding any model's context window, and in a way that works across
providers and even small/cheap models. Three forces shape the design:

1. **Token limits** → we cannot put the whole repo in one prompt → *map-reduce*.
2. **Consistency / machine-readability** → the model output must obey a fixed schema → *Pydantic-validated structured output*.
3. **Model-agnosticism** → no hard dependency on one vendor, and graceful behavior on weak models → *pluggable LLM factory + layered structured-output fallback*.

## 2. Pipeline overview

```
repo ──▶ load & filter ──▶ plan work units ──▶ MAP (per file) ──▶ REDUCE (hierarchical) ──▶ assemble ──▶ JSON
        repo_loader.py      chunker.py          extractor.py        synthesizer.py            analyzer.py
        (LlamaIndex)        (token budget)      (LLM, parallel)     (LLM, code-free facts)    (+ deterministic)
```

## 3. Module responsibilities

| Module | Responsibility | Key decisions |
|---|---|---|
| `config.py` | Env-driven settings, all with defaults | No LLM imports here, so `--dry-run` runs on the minimal dependency set. |
| `token_utils.py` | Provider-neutral token counting | `tiktoken cl100k_base` as a cross-model proxy; char/4 fallback if unavailable. |
| `repo_loader.py` | Discover, load, classify, rank files | LlamaIndex `SimpleDirectoryReader` for loading; we control selection (extensions, excluded dirs, **importance ranking**, **skip oversized data dumps**). |
| `chunker.py` | Turn files into token-budgeted work units | Fits-in-budget → 1 unit; too big → **syntactic** split via LlamaIndex `CodeSplitter` (tree-sitter), else token split. |
| `schemas.py` | The output contract (Pydantic) | Map schema (`ComponentAnalysis`) vs reduce schemas (`ProjectOverview`, `ComplexityAnalysis`, …). Enums keep values consistent. |
| `structured.py` | Get valid JSON out of *any* model | 3 layers: native tool-calling → `PydanticOutputParser` → one JSON-repair round-trip. |
| `llm_factory.py` | Build the chat model from config | LangChain `init_chat_model(provider, model)`; OpenAI-compatible `base_url` path for local servers. |
| `prompts.py` | Map & reduce prompt templates | Instruction-dense, "facts only, don't invent" — improves valid-output rate on small models. |
| `extractor.py` | The **map** pass | Concurrent per-unit LLM calls; merges chunk results back into one component; records which structured-output mode was used. |
| `synthesizer.py` | The **reduce** pass | Groups components by domain → domain summaries → project overview; **never re-sends source code**, only compact facts. |
| `analyzer.py` | Orchestration + deterministic assembly | Endpoint catalog, statistics, build signals computed in Python — no tokens wasted on what a parser can do reliably. |
| `main.py` | CLI | `--dry-run` (no API key), `--repo-url/--repo-path`, `--max-files`, `--output`. |

## 4. Why map-reduce (the core idea)

- **Map** is *embarrassingly parallel*: each file is independent, so per-file extraction runs
  concurrently and each request is tiny (one file, well under budget). Files larger than the budget
  are split on method/class boundaries so each chunk is still meaningful; the per-chunk
  `ComponentAnalysis` objects are then merged (union of methods/deps/notes, max of complexity).
- **Reduce** is *hierarchical* to stay cheap and bounded: `components → per-domain summaries →
  project overview`. Crucially, the reduce prompts contain only the **compact structured facts**
  from the map pass (class name, role, method names, complexity) — **never the source code again**.
  That is what keeps synthesis within budget no matter how large the repo is.

## 5. Why the 3-layer structured output matters

Not every model supports tool/function calling (many small or local models don't). So:

1. **native** — `llm.with_structured_output(schema)`: best on capable models.
2. **parser** — inject the JSON schema as format instructions, then `PydanticOutputParser` parses the
   raw text. Works with *zero* tool-calling support.
3. **repair** — if parsing fails, strip markdown fences / extract the outer `{...}`, and if still
   invalid, do a single "fix this into valid JSON" round-trip.

`STRUCTURED_OUTPUT_MODE=auto` tries native and silently falls back. This is the concrete mechanism
behind "works with even the cheapest models."

## 6. Best-practice decisions (and the trade-offs)

- **Skip data, not code.** Sakila ships an 850k-token SQL *data* dump; analyzing it would blow the
  budget for zero comprehension value, so oversized `.sql/.csv/.json` files are skipped. Trade-off:
  a genuinely huge *source* file is chunked, not skipped.
- **Determinism.** `temperature=0` for repeatable, reviewable output.
- **Compute, don't prompt.** The REST endpoint catalog and statistics are derived in Python from the
  extracted methods — accurate and free of token cost/hallucination.
- **Graceful degradation.** LlamaIndex, tree-sitter, and tiktoken each have fallbacks, so the tool
  runs in a minimal environment and never hard-fails on an optional dependency.
- **Cost controls.** `MAX_FILES` + importance ranking (controllers > services > repositories > …)
  means a quick/cheap run still covers the most informative files first.

## 7. How correctness was verified without an API key

- `--dry-run` exercises the entire non-LLM pipeline (discovery, chunking, budgeting) against the real
  repo: 210 files → 215 work units, ~121k input tokens estimated.
- `tests/test_smoke.py` builds a tiny synthetic Java repo and runs the **full** map-reduce with a mock
  LLM, asserting a schema-valid `CodebaseAnalysis` that round-trips through JSON.
- The committed `output/sakila_analysis.json` is constructed *through* the same Pydantic schemas, so it
  is guaranteed schema-conformant, and its endpoint catalog is derived directly from the controllers'
  mapping annotations.

## 8. Known limitations & natural extensions

- Token counting is approximate across non-OpenAI tokenizers (budgets set conservatively).
- Per-file granularity captures dependencies qualitatively, not as a precise call graph.
- Natural next steps: batch multiple small files per request to cut call count; build a LlamaIndex
  retrieval index over the components for interactive Q&A; add per-method cyclomatic-complexity metrics
  computed statically to complement the LLM's qualitative rating.
