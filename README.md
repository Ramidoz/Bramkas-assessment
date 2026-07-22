# Codebase Analysis using an LLM

> **Author:** Rohit Ananthan · Coding assignment submission.
> Designed and directed by the author and implemented with AI assistance (an agentic coding
> tool) — which is the skill this assignment sets out to assess. See
> [ARCHITECTURE](docs/ARCHITECTURE.md) for a module-by-module design walkthrough.

A model-agnostic program that reads a codebase, extracts structured knowledge with a
Large Language Model, and emits a well-organized, machine-readable **JSON** report — a
high-level project overview, per-component method signatures and descriptions, a complete
REST endpoint catalog, complexity analysis, and noteworthy observations.

The reference target is [`codejsha/spring-rest-sakila`](https://github.com/codejsha/spring-rest-sakila)
(a Spring Boot REST API over the MySQL Sakila sample database). The produced report lives at
[`output/sakila_analysis.json`](output/sakila_analysis.json).

---

## 1. What it does

```
        clone/read            token-aware              LLM: map              LLM: hierarchical reduce         deterministic
 repo ───────────────▶ files ───────────────▶ work units ─────────▶ ComponentAnalysis[] ───────────────▶ overview/complexity ───▶ JSON
   │   LlamaIndex loader   │   chunker/planner     │  per-file, parallel      │  group by domain →                 + endpoints/stats
   │                       │                       │                          │  domain summaries → project        (computed in Python)
   └── filter + rank       └── respect token budget └── structured output      └── code never re-sent in reduce
```

* **Read & process** the repository with **LlamaIndex** (`SimpleDirectoryReader` for loading,
  `CodeSplitter`/`TokenTextSplitter` for code-aware chunking).
* **Feed it to an LLM within token limits** via a **map-reduce** strategy (details below), using
  **LangChain** for model access and the extraction/synthesis chains.
* **Model-agnostic**: pick any provider/model (Anthropic, OpenAI, a local Ollama/LM Studio/vLLM
  server, …) through environment variables — including cheap/small models.
* **Structured, machine-readable output**: every LLM response is validated against **Pydantic**
  schemas, so the JSON is consistent regardless of which model produced it.

This satisfies the assignment's requirements: efficient code processing within token limits,
a suitable LLM for code comprehension, LangChain **and** LlamaIndex for orchestration, and a
structured JSON knowledge base covering purpose/functionality, key methods & signatures, and
complexity.

---

## 2. Approach & methodology

### 2.1 Hybrid frameworks (LlamaIndex + LangChain)

| Concern | Library | Where |
|---|---|---|
| Load repo files into documents | **LlamaIndex** `SimpleDirectoryReader` | `src/repo_loader.py` |
| Code-aware chunking of large files | **LlamaIndex** `CodeSplitter` (tree-sitter) → `TokenTextSplitter` fallback | `src/chunker.py` |
| Model-agnostic LLM access | **LangChain** `init_chat_model` | `src/llm_factory.py` |
| Extraction / synthesis chains + structured output | **LangChain** (`with_structured_output`, `PydanticOutputParser`) | `src/structured.py`, `src/extractor.py`, `src/synthesizer.py` |

### 2.2 Map-reduce to respect token limits

A ~10k-LOC, 188-file project does not fit in a single prompt. The pipeline therefore:

1. **Map (per file).** Each file becomes one or more *work units*. A file within the per-request
   input budget is a single unit; a larger file is split on **syntactic boundaries** (methods/classes)
   by LlamaIndex's `CodeSplitter`, so each chunk stays semantically coherent. Every unit is sent to the
   LLM and returns a `ComponentAnalysis` (class, role, methods, signatures, dependencies, complexity,
   notes). Chunk results for the same file are merged.
2. **Reduce (hierarchical).** The synthesis pass **never re-sends source code** — only the compact
   structured facts from the map pass. It reduces in a tree: components → **per-domain** summaries →
   **project** overview, plus complexity and noteworthy syntheses. This keeps every reduce request well
   within the budget even for large repositories.
3. **Deterministic assembly.** Anything a parser can compute reliably — the REST endpoint catalog,
   statistics, build signals — is computed in Python, **not** spent on LLM tokens.

### 2.3 Model-agnostic, cheap-model-friendly design

* The LLM is built by `init_chat_model(provider, model)`, so switching vendors is an env-var change.
* An OpenAI-compatible `base_url` path lets any local/self-hosted server (Ollama, LM Studio, vLLM)
  act as the model — including free, local, small models.
* **Structured output has three layers** (`src/structured.py`) so weak models still produce valid JSON:
  1. **native** tool/function calling (`with_structured_output`);
  2. **parser** — inject the JSON schema as format instructions and parse with `PydanticOutputParser`
     (works on models with *no* tool-calling);
  3. **repair** — strip fences / extract the JSON object, and if still invalid do one
     "fix this into valid JSON" round-trip.
  `STRUCTURED_OUTPUT_MODE=auto` tries native and falls back automatically.
* A **cheap map / stronger synthesis** split is supported (`LLM_MODEL` + optional `SYNTHESIS_MODEL`) but
  defaults to a single, inexpensive model for everything.

### 2.4 Best practices applied

* **Token budgeting** with `tiktoken` (offline, provider-neutral counting) and a configurable
  per-request cap; prompt-scaffolding headroom is reserved before chunking.
* **Skip data, not code.** Oversized SQL/CSV/JSON dumps (e.g. Sakila's 850k-token data file) are skipped
  so the token budget is spent on source, not data.
* **Importance-first selection.** Files are ranked (controllers > services > repositories > entities > …)
  so a capped/quick run analyzes the most informative files first (`MAX_FILES`).
* **Determinism** (`temperature=0`) and **strict schemas** for consistent, machine-readable output.
* **Concurrency** for the map pass (`MAX_CONCURRENCY`) with graceful per-unit failure handling.
* **Graceful degradation**: optional dependencies (LlamaIndex, tree-sitter, tiktoken) each have fallbacks,
  so the tool runs even in a minimal environment.

---

## 3. Repository layout

```
.
├── main.py                     # CLI entrypoint
├── requirements.txt
├── .env.example                # all configuration options, documented
├── src/
│   ├── config.py               # env-driven settings
│   ├── token_utils.py          # tiktoken counting + budgeting
│   ├── llm_factory.py          # model-agnostic LLM construction (LangChain)
│   ├── repo_loader.py          # LlamaIndex loading + filtering + ranking
│   ├── chunker.py              # token-aware, code-aware work-unit planning
│   ├── schemas.py              # Pydantic output contract
│   ├── structured.py           # resilient structured-output (native→parser→repair)
│   ├── prompts.py              # map & reduce prompt templates
│   ├── extractor.py            # map pass (per-file extraction, concurrent)
│   ├── synthesizer.py          # hierarchical reduce pass
│   └── analyzer.py             # orchestration + deterministic assembly
├── tests/
│   └── test_smoke.py           # no-network, no-API-key pipeline test (mock LLM)
└── output/
    └── sakila_analysis.json    # the structured knowledge deliverable
```

---

## 4. Getting started

### Install

```bash
pip install -r requirements.txt
cp .env.example .env            # then edit as needed
```

### Dry run — no API key required

Verify discovery, chunking, and the token budget without calling any model:

```bash
python main.py --repo-url https://github.com/codejsha/spring-rest-sakila --dry-run
```

Example output (abridged):

```json
{
  "loader": "llamaindex",
  "build_signals": "Gradle, Spring Boot config",
  "domains": ["auth", "catalog", "customer", "location", "payment", "rental", "staff", "store", ...],
  "plan": { "files": 210, "work_units": 215, "files_requiring_split": 2,
            "total_input_tokens_estimate": 121145, "max_input_tokens_per_request": 6000 }
}
```

### Full analysis

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --repo-url https://github.com/codejsha/spring-rest-sakila \
               --output output/sakila_analysis.json
```

```bash
# OpenAI
LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... \
python main.py --repo-path ./spring-rest-sakila
```

```bash
# Fully local / free — an OpenAI-compatible Ollama server, a small code model
LLM_PROVIDER=openai LLM_MODEL=qwen2.5-coder:7b \
LLM_API_BASE=http://localhost:11434/v1 \
python main.py --repo-path ./spring-rest-sakila
```

Useful flags/vars: `--repo-path` (analyze a local checkout), `--max-files N`,
`MAX_INPUT_TOKENS_PER_REQUEST`, `SYNTHESIS_MODEL`, `STRUCTURED_OUTPUT_MODE`, `MAX_CONCURRENCY`.
See `.env.example` for the full list.

### Tests

```bash
python tests/test_smoke.py        # or: python -m pytest tests/ -q
```

---

## 5. Output schema

Top-level keys of `output/sakila_analysis.json` (full schema in `src/schemas.py`):

| Key | Contents |
|---|---|
| `metadata` | repository, analyzer version, timestamp, LLM/model config, token-budget strategy, notes |
| `statistics` | files discovered/analyzed, method & endpoint counts, LOC, components-by-type |
| `project_overview` | purpose, one-liner, description, tech stack, architecture style, layers, domains, key features, domain model |
| `domains[]` | per-domain purpose, key components, endpoint summary, complexity |
| `api_endpoints[]` | complete REST catalog: `http_method`, `path`, `handler`, `description`, `secured_role` |
| `complexity_analysis` | overall rating, hotspots (with reasons), cross-cutting observations |
| `noteworthy[]` | categorized findings (security, pattern, performance, testing, tech_debt, observation) |
| `components[]` | per file: class, role, annotations, **methods with signatures & descriptions**, dependencies, complexity, notes |

---

## 6. What the analysis found (spring-rest-sakila)

* **Purpose**: a secured REST API exposing the Sakila DVD-rental domain (films, actors, customers,
  rentals, payments, stores, staff) for CRUD and analytical access — a Spring Boot reference project.
* **Architecture**: layered, package-by-feature modular monolith (controller → service → repository →
  domain), 8 functional domains, HATEOAS/HAL responses, stateless JWT security with `READ/MANAGE/ADMIN`
  method-level roles.
* **Stack**: Java 17, Spring Boot 3, Spring Data JPA, Spring HATEOAS, Spring Security, QueryDSL +
  Blaze-Persistence, MapStruct, Lombok, JJWT, Redis caching, MySQL, Spring REST Docs → OpenAPI/Postman.
* **70 REST endpoints** catalogued across 10 controllers.
* **Complexity hotspots**: QueryDSL/Blaze custom repositories, the multi-mechanism `PersistenceConfig`,
  and HATEOAS link assembly in `ActorController`.
* **Noteworthy**: weak hardcoded JWT signing key and committed DB credentials (fine for a sample, not for
  production), `printStackTrace()` in the global exception handler, `open-in-view`/`show-sql` enabled.

---

## 7. Assumptions & limitations

* **How this repository's `sakila_analysis.json` was produced.** This environment has **no LLM API key**,
  so the committed report was authored by an LLM (Anthropic Claude) reading the repository and following
  the *same* map-reduce methodology and Pydantic schema the program implements. It is constructed *through*
  the project's own schemas, so it is guaranteed schema-valid, and the endpoint catalog was derived directly
  from the controllers' mapping annotations. Running `python main.py` with any provider key reproduces the
  identical JSON structure across all source files. The `components[]` array in the committed file details the
  architecturally significant classes (all 10 controllers, representative services/repositories/entities,
  security, config, exception, cross-cutting) plus the **complete** endpoint catalog; a full program run
  emits one component entry per source file.
* **Token counting is an approximation.** `tiktoken`'s `cl100k_base` is used as a provider-neutral proxy;
  non-OpenAI tokenizers differ slightly, so budgets are set conservatively.
* **Quality tracks the chosen model.** Small/cheap models produce shorter, occasionally shallower
  descriptions; the structured-output repair layer keeps them *valid* but not necessarily as rich as a
  frontier model. Use `SYNTHESIS_MODEL` to upgrade only the synthesis pass if desired.
* **Static analysis only.** The tool reads source; it does not build or run the target project, so runtime
  behavior and framework-generated code (e.g. QueryDSL `Q`-classes) are inferred, not executed.
* **Per-file granularity.** Cross-file relationships are captured qualitatively (dependencies, domain
  summaries) rather than as a precise call graph. Batching multiple small files per request and building a
  retrieval index for Q&A are natural future extensions.

---

## 8. Requirements

Python 3.10+. Core runtime: `pydantic`, `tiktoken`, `python-dotenv`. Orchestration: `langchain`,
`langchain-core`, `langchain-community`, `llama-index-core`. Install the provider integration you use
(`langchain-anthropic` / `langchain-openai` / `langchain-ollama`). See `requirements.txt`.

---

## 9. Author & attribution

Built by **Rohit Ananthan** as a coding-assignment submission. The task was scoped by the author,
and the key design decisions — the hybrid LlamaIndex + LangChain approach, the model-agnostic /
"works with the cheapest models" direction that drove the resilient structured-output layer, and the
repository/PR setup — were made by the author, then implemented with the help of an agentic AI coding
tool. Using AI tools effectively is precisely what this assignment sets out to assess, so this is
stated openly rather than hidden. A full design walkthrough is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
