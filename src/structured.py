"""Resilient structured-output extraction.

Getting *valid, schema-conforming* JSON out of an arbitrary model — including
small/cheap/local ones that don't support tool-calling — is the trickiest part
of a model-agnostic design. This module implements a layered strategy:

1. **native**  — ``llm.with_structured_output(schema)`` (tool/function calling).
   Fast and reliable on capable models.
2. **parser**  — inject the JSON schema as format instructions into the prompt,
   call the model, then parse with a Pydantic parser. Works on models with no
   tool-calling at all.
3. **repair**  — if parsing fails, strip markdown fences / extract the outer JSON
   object, and if still invalid do a single "fix this into valid JSON" round-trip.

``mode="auto"`` tries native first and falls back to parser+repair. This keeps
the pipeline running on the widest possible range of models.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _text_of(response) -> str:
    """Extract plain text from a LangChain message or raw string."""
    if response is None:
        return ""
    content = getattr(response, "content", response)
    if isinstance(content, list):  # some providers return content blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a single JSON object from noisy model output."""
    if not text:
        return text
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


class StructuredResult:
    """A parsed value plus provenance (which strategy produced it)."""

    def __init__(self, value: Optional[BaseModel], mode: str, error: Optional[str] = None):
        self.value = value
        self.mode = mode
        self.error = error

    @property
    def ok(self) -> bool:
        return self.value is not None


def extract_structured(
    llm,
    schema: Type[T],
    system_prompt: str,
    human_prompt: str,
    mode: str = "auto",
) -> StructuredResult:
    """Call `llm` and return an instance of `schema` (or a failed result).

    `mode` is one of ``auto`` | ``native`` | ``parser``.
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    # ---- 1. native structured output ------------------------------------- #
    if mode in ("auto", "native"):
        try:
            structured_llm = llm.with_structured_output(schema)
            value = structured_llm.invoke(messages)
            if isinstance(value, schema):
                return StructuredResult(value, "native")
            # some providers return a dict
            return StructuredResult(schema.model_validate(value), "native")
        except Exception as exc:  # noqa: BLE001 - fall through to parser
            if mode == "native":
                return StructuredResult(None, "native", error=str(exc))

    # ---- 2. parser path (prompt-injected schema) ------------------------- #
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=schema)
    format_instructions = parser.get_format_instructions()
    parser_human = (
        f"{human_prompt}\n\n"
        f"Return ONLY a JSON object that conforms to this schema. "
        f"Do not include explanations or markdown fences.\n\n{format_instructions}"
    )
    parser_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=parser_human),
    ]

    try:
        raw = llm.invoke(parser_messages)
    except Exception as exc:  # noqa: BLE001
        return StructuredResult(None, "parser", error=f"invoke failed: {exc}")

    text = _text_of(raw)

    # 2a. direct parse
    for candidate in (text, _extract_json_object(text)):
        try:
            return StructuredResult(parser.parse(candidate), "parser")
        except Exception:
            continue
    # 2b. try plain json load on the extracted object
    try:
        obj = json.loads(_extract_json_object(text))
        return StructuredResult(schema.model_validate(obj), "parser")
    except Exception:
        pass

    # ---- 3. one repair round-trip ---------------------------------------- #
    repair_human = (
        "The following text was supposed to be a JSON object matching the schema "
        "below but could not be parsed. Fix it and return ONLY valid JSON.\n\n"
        f"SCHEMA:\n{format_instructions}\n\nTEXT TO FIX:\n{text}"
    )
    try:
        fixed = llm.invoke(
            [SystemMessage(content="You repair malformed JSON."), HumanMessage(content=repair_human)]
        )
        fixed_text = _extract_json_object(_text_of(fixed))
        return StructuredResult(parser.parse(fixed_text), "parser+repair")
    except Exception as exc:  # noqa: BLE001
        return StructuredResult(None, "parser+repair", error=f"unparseable: {exc}")
