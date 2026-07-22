"""Codebase Analyzer — LLM-powered structured knowledge extraction.

A model-agnostic pipeline that reads a codebase, extracts structured knowledge
with an LLM (map-reduce over files to respect token limits), and emits a
machine-readable JSON report.

Frameworks (hybrid):
    * LlamaIndex — repository loading and code-aware chunking
    * LangChain  — model-agnostic LLM access + extraction/synthesis chains
"""

__version__ = "1.0.0"
