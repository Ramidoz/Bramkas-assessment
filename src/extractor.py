"""Map pass — extract a :class:`ComponentAnalysis` for each source file.

Each work unit is sent to the LLM with the map prompt and the resilient
structured-output helper. Files that were split into multiple chunks have their
partial results merged back into a single component record.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

from .chunker import WorkUnit
from .config import Settings
from .prompts import MAP_HUMAN, MAP_SYSTEM
from .schemas import ComplexityRating, ComponentAnalysis, ComponentType
from .structured import extract_structured


@dataclass
class MapStats:
    llm_calls: int = 0
    failures: int = 0
    modes: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.modes is None:
            self.modes = {}

    def record(self, mode: str, ok: bool):
        self.llm_calls += 1
        self.modes[mode] = self.modes.get(mode, 0) + 1
        if not ok:
            self.failures += 1


def _analyze_unit(llm, unit: WorkUnit, settings: Settings) -> tuple[Optional[ComponentAnalysis], str]:
    chunk_note = ""
    if unit.is_partial:
        chunk_note = (
            f"NOTE: This is chunk {unit.chunk_index + 1} of {unit.chunk_count} of a "
            f"large file; analyze only what is present here."
        )
    human = MAP_HUMAN.format(
        file_path=unit.file.rel_path,
        type_hint=unit.file.type_hint,
        chunk_note=chunk_note,
        code=unit.text,
    )
    result = extract_structured(
        llm, ComponentAnalysis, MAP_SYSTEM, human, mode=settings.structured_output_mode
    )
    comp = result.value
    if comp is not None:
        comp.file_path = unit.file.rel_path  # trust the real path over the model
    return comp, result.mode


_COMPLEXITY_ORDER = {
    ComplexityRating.low: 0,
    ComplexityRating.medium: 1,
    ComplexityRating.high: 2,
}


def _merge_components(parts: List[ComponentAnalysis]) -> ComponentAnalysis:
    """Merge chunk-level analyses of the same file into one component record."""
    if len(parts) == 1:
        return parts[0]

    base = parts[0].model_copy(deep=True)
    seen_methods = {(m.name, m.signature) for m in base.methods}
    for extra in parts[1:]:
        base.class_name = base.class_name or extra.class_name
        base.package = base.package or extra.package
        if base.component_type == ComponentType.other and extra.component_type != ComponentType.other:
            base.component_type = extra.component_type
        for m in extra.methods:
            key = (m.name, m.signature)
            if key not in seen_methods:
                seen_methods.add(key)
                base.methods.append(m)
        for dep in extra.dependencies:
            if dep not in base.dependencies:
                base.dependencies.append(dep)
        for ann in extra.annotations:
            if ann not in base.annotations:
                base.annotations.append(ann)
        for note in extra.noteworthy:
            if note not in base.noteworthy:
                base.noteworthy.append(note)
        if _COMPLEXITY_ORDER[extra.complexity] > _COMPLEXITY_ORDER[base.complexity]:
            base.complexity = extra.complexity
    return base


def run_map(
    llm,
    units: List[WorkUnit],
    settings: Settings,
    progress=None,
) -> tuple[List[ComponentAnalysis], MapStats]:
    """Run the map pass over all work units, returning merged components."""
    stats = MapStats()
    by_file: Dict[str, List[ComponentAnalysis]] = {}
    order: List[str] = []

    max_workers = max(1, settings.max_concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_analyze_unit, llm, unit, settings): unit for unit in units
        }
        done = 0
        for fut in as_completed(future_map):
            unit = future_map[fut]
            try:
                comp, mode = fut.result()
                stats.record(mode, comp is not None)
            except Exception as exc:  # noqa: BLE001
                comp, mode = None, "error"
                stats.record("error", False)
            done += 1
            if progress:
                progress(done, len(units), unit.file.rel_path)
            if comp is None:
                continue
            if unit.file.rel_path not in by_file:
                by_file[unit.file.rel_path] = []
                order.append(unit.file.rel_path)
            by_file[unit.file.rel_path].append(comp)

    components: List[ComponentAnalysis] = []
    for rel in order:
        parts = sorted(by_file[rel], key=lambda c: 0)  # keep insertion order
        components.append(_merge_components(parts))
    return components, stats
