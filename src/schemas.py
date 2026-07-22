"""Pydantic schemas — the machine-readable contract for LLM output.

Two families:

* **Map schemas** (``ComponentAnalysis`` and friends) — what the LLM returns for
  a single file/component.
* **Reduce / report schemas** (``ProjectOverview``, ``ComplexityAnalysis``,
  ``CodebaseAnalysis``) — the synthesised, whole-project view and the final
  top-level document.

Enforcing these schemas is what keeps the output consistent and machine-readable
regardless of which model produced it.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ComplexityRating(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ComponentType(str, Enum):
    controller = "controller"
    service = "service"
    repository = "repository"
    entity = "entity"
    dto = "dto"
    mapper = "mapper"
    assembler = "assembler"
    converter = "converter"
    config = "config"
    security = "security"
    filter = "filter"
    exception = "exception"
    constant = "constant"
    util = "util"
    serializer = "serializer"
    application = "application"
    other = "other"


# --------------------------------------------------------------------------- #
# Map-level schemas (per file/component)
# --------------------------------------------------------------------------- #
class MethodInfo(BaseModel):
    name: str = Field(description="Method or function name.")
    signature: str = Field(
        description="Full signature: return type, name, and parameters."
    )
    description: str = Field(
        description="One or two sentences on what the method does."
    )
    http_method: Optional[str] = Field(
        default=None,
        description="HTTP verb (GET/POST/PUT/DELETE/PATCH) if this is a REST handler.",
    )
    http_path: Optional[str] = Field(
        default=None,
        description="Route path if this is a REST handler (relative to the class mapping).",
    )
    parameters: List[str] = Field(
        default_factory=list, description="Parameter names/types."
    )
    returns: Optional[str] = Field(default=None, description="What the method returns.")
    complexity: ComplexityRating = Field(
        default=ComplexityRating.low,
        description="Estimated complexity of this single method.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Noteworthy details: caching, transactions, security, side effects.",
    )


class ComponentAnalysis(BaseModel):
    """LLM analysis of a single source file / class."""

    file_path: str = Field(description="Repository-relative path of the file.")
    class_name: Optional[str] = Field(
        default=None, description="Primary class/interface/enum name."
    )
    package: Optional[str] = Field(default=None, description="Java package (if any).")
    component_type: ComponentType = Field(
        default=ComponentType.other,
        description="Architectural role of this component.",
    )
    responsibility: str = Field(
        description="A concise statement of what this component is responsible for."
    )
    annotations: List[str] = Field(
        default_factory=list,
        description="Key framework annotations present (e.g. @RestController, @Service).",
    )
    methods: List[MethodInfo] = Field(
        default_factory=list,
        description="Notable methods with signatures and descriptions.",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Important collaborators/types this component depends on.",
    )
    complexity: ComplexityRating = Field(
        default=ComplexityRating.low,
        description="Overall complexity rating for the component.",
    )
    noteworthy: List[str] = Field(
        default_factory=list,
        description="Patterns, potential issues, security/perf notes worth flagging.",
    )


# --------------------------------------------------------------------------- #
# Reduce-level schemas (whole project)
# --------------------------------------------------------------------------- #
class DomainSummary(BaseModel):
    """Intermediate summary for one functional domain/package (hierarchical reduce)."""

    domain: str = Field(description="Domain/package name (e.g. catalog, rental).")
    purpose: str = Field(description="What this domain is responsible for.")
    key_components: List[str] = Field(
        default_factory=list, description="Most important classes in the domain."
    )
    endpoints_summary: Optional[str] = Field(
        default=None, description="Short summary of REST endpoints exposed, if any."
    )
    complexity: ComplexityRating = Field(default=ComplexityRating.low)
    notes: List[str] = Field(default_factory=list)


class ProjectOverview(BaseModel):
    name: str = Field(description="Project name.")
    one_liner: str = Field(description="A single sentence describing the project.")
    purpose: str = Field(description="The project's purpose and problem it solves.")
    description: str = Field(
        description="A richer paragraph on functionality and how it is structured."
    )
    tech_stack: List[str] = Field(
        default_factory=list, description="Languages, frameworks, notable libraries."
    )
    architecture_style: str = Field(
        description="Architectural style (e.g. layered REST, hexagonal, microservice)."
    )
    layers: List[str] = Field(
        default_factory=list,
        description="Architectural layers present (controller/service/repository/...).",
    )
    domains: List[str] = Field(
        default_factory=list, description="Functional domains/modules."
    )
    key_features: List[str] = Field(
        default_factory=list, description="Headline capabilities of the system."
    )
    domain_model_summary: str = Field(
        default="",
        description="Summary of the core domain/data model the app operates on.",
    )


class ComplexityHotspot(BaseModel):
    component: str = Field(description="File or class that is a complexity hotspot.")
    reason: str = Field(description="Why it is complex.")
    rating: ComplexityRating = Field(default=ComplexityRating.high)


class ComplexityAnalysis(BaseModel):
    overall_rating: ComplexityRating = Field(default=ComplexityRating.medium)
    summary: str = Field(description="Narrative on the codebase's overall complexity.")
    hotspots: List[ComplexityHotspot] = Field(default_factory=list)
    observations: List[str] = Field(
        default_factory=list,
        description="Cross-cutting complexity observations (coupling, patterns, etc.).",
    )


class NoteworthyCategory(str, Enum):
    pattern = "pattern"
    security = "security"
    performance = "performance"
    testing = "testing"
    tech_debt = "tech_debt"
    observation = "observation"


class NoteworthyItem(BaseModel):
    category: NoteworthyCategory = Field(default=NoteworthyCategory.observation)
    title: str = Field(description="Short title of the observation.")
    detail: str = Field(description="Explanation of the observation.")


class NoteworthyList(BaseModel):
    """Wrapper so the LLM can return a list of noteworthy items in one call."""

    items: List[NoteworthyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Deterministic (code-computed) pieces — not produced by the LLM
# --------------------------------------------------------------------------- #
class ApiEndpoint(BaseModel):
    http_method: str
    path: str
    handler: str = Field(description="Controller.method that serves this endpoint.")
    description: Optional[str] = None
    secured_role: Optional[str] = None


class RunMetadata(BaseModel):
    repository: str
    analyzer_version: str
    generated_at: Optional[str] = None
    llm: dict = Field(default_factory=dict)
    token_budget: dict = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class Statistics(BaseModel):
    files_discovered: int = 0
    files_analyzed: int = 0
    total_methods: int = 0
    total_endpoints: int = 0
    lines_of_code: int = 0
    llm_calls: int = 0
    components_by_type: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Top-level document
# --------------------------------------------------------------------------- #
class CodebaseAnalysis(BaseModel):
    """The complete structured knowledge document written to JSON."""

    metadata: RunMetadata
    statistics: Statistics
    project_overview: Optional[ProjectOverview] = None
    domains: List[DomainSummary] = Field(default_factory=list)
    api_endpoints: List[ApiEndpoint] = Field(default_factory=list)
    complexity_analysis: Optional[ComplexityAnalysis] = None
    noteworthy: List[NoteworthyItem] = Field(default_factory=list)
    components: List[ComponentAnalysis] = Field(default_factory=list)
