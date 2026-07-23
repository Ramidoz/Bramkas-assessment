"""Self-contained smoke tests — no network, no API key.

Creates a tiny synthetic Java repo in a temp dir and exercises the full
pipeline (discover → plan → map → reduce → assemble → serialize) using a mock
LLM. Run with:  python -m pytest tests/ -q   (or)  python tests/test_smoke.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import analyzer, llm_factory
from src.config import load_settings
from src.schemas import (
    CodebaseAnalysis, ComplexityAnalysis, ComplexityRating as CR, ComponentAnalysis,
    ComponentType as CT, DomainSummary, MethodInfo, NoteworthyItem, NoteworthyList,
    ProjectOverview,
)

CONTROLLER = """
package com.example.app.services.demo.controller;

@RestController
@RequestMapping("/widgets")
public class WidgetController {
    @GetMapping(path = "")
    public ResponseEntity<List<Widget>> list() { return ResponseEntity.ok(service.all()); }

    @PostMapping(path = "")
    public ResponseEntity<Void> create(@RequestBody WidgetRequest r) { return ResponseEntity.created(...).build(); }
}
"""

SERVICE = """
package com.example.app.services.demo.service;

@Service
public class WidgetServiceImpl implements WidgetService {
    public List<Widget> all() { return repo.findAll(); }
}
"""


def _make_repo(root: str) -> None:
    base = os.path.join(root, "src/main/java/com/example/app/services/demo")
    os.makedirs(os.path.join(base, "controller"))
    os.makedirs(os.path.join(base, "service"))
    with open(os.path.join(base, "controller", "WidgetController.java"), "w") as fh:
        fh.write(CONTROLLER)
    with open(os.path.join(base, "service", "WidgetServiceImpl.java"), "w") as fh:
        fh.write(SERVICE)
    with open(os.path.join(root, "build.gradle.kts"), "w") as fh:
        fh.write("plugins { id(\"java\") }\n")


class _Structured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        s = self.schema
        if s is ComponentAnalysis:
            return ComponentAnalysis(
                file_path="x", class_name="WidgetController", component_type=CT.controller,
                responsibility="Handles widget endpoints.",
                methods=[MethodInfo(name="list", signature="ResponseEntity<List<Widget>> list()",
                                    description="lists widgets", http_method="GET", http_path="/widgets")],
                complexity=CR.low)
        if s is DomainSummary:
            return DomainSummary(domain="demo", purpose="demo domain", complexity=CR.low)
        if s is ProjectOverview:
            return ProjectOverview(name="demo", one_liner="demo", purpose="p", description="d",
                                   architecture_style="layered REST")
        if s is ComplexityAnalysis:
            return ComplexityAnalysis(summary="ok")
        if s is NoteworthyList:
            return NoteworthyList(items=[NoteworthyItem(title="t", detail="d")])
        raise AssertionError(f"unexpected schema {s}")


class MockLLM:
    def with_structured_output(self, schema):
        return _Structured(schema)

    def invoke(self, messages):
        class _M:
            content = "{}"
        return _M()


def test_prepare_and_dry_run():
    with tempfile.TemporaryDirectory() as root:
        _make_repo(root)
        settings = load_settings()
        files, units, counter = analyzer.prepare(root, settings)
        assert len(files) >= 2, "should discover the java files"
        assert any(f.type_hint == "controller" for f in files)
        report = analyzer.dry_run(root, settings)
        assert report["plan"]["files"] == len(files)
        assert "Gradle" in report["build_signals"]
        print("test_prepare_and_dry_run: OK", report["plan"])


def test_full_pipeline_with_mock_llm():
    with tempfile.TemporaryDirectory() as root:
        _make_repo(root)
        settings = load_settings()
        llm_factory.build_map_llm = lambda s, **k: MockLLM()
        llm_factory.build_synthesis_llm = lambda s, **k: MockLLM()
        result = analyzer.analyze(root, settings, repository_name="demo")
        assert isinstance(result, CodebaseAnalysis)
        assert result.project_overview is not None
        assert result.statistics.total_endpoints >= 1
        # must round-trip through JSON cleanly
        payload = json.dumps(result.model_dump(mode="json"))
        assert len(payload) > 100
        print("test_full_pipeline_with_mock_llm: OK",
              "endpoints=", result.statistics.total_endpoints,
              "components=", len(result.components))


if __name__ == "__main__":
    test_prepare_and_dry_run()
    test_full_pipeline_with_mock_llm()
    print("\nAll smoke tests passed.")
