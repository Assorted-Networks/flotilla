"""Team and cluster configuration (the YAML file mounted into the coordinator)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

STRATEGIES = ("single", "mixture", "council", "critique", "plan", "vote", "route", "pipeline")


class ConfigError(Exception):
    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


class Params(BaseModel):
    """Generation parameters shared by `defaults` and every member."""

    model_config = ConfigDict(extra="forbid")

    system: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    # Passed through to the backend. Ollama maps "none" to "thinking off",
    # which keeps small thinking models (qwen3.5, gemma4, ...) fast.
    reasoning_effort: Optional[str] = None
    timeout: Optional[float] = Field(default=None, gt=0)
    # Placement: pin to a node name, require labels, or prefer labels.
    node: Optional[str] = None
    labels: dict[str, str] = Field(default_factory=dict)
    prefer_labels: dict[str, str] = Field(default_factory=dict)
    # Anything else merged into the request body verbatim, e.g.
    # {"chat_template_kwargs": {"enable_thinking": false}} for vLLM.
    extra: dict[str, Any] = Field(default_factory=dict)


class MemberConfig(Params):
    """One team member: a model (with fallbacks) plus a role, or a nested team."""

    model: Optional[Union[str, list[str]]] = None
    team: Optional[str] = None
    description: str = ""

    @model_validator(mode="after")
    def _one_target(self) -> "MemberConfig":
        if bool(self.model) == bool(self.team):
            raise ValueError("a member needs exactly one of `model` or `team`")
        return self

    @property
    def candidates(self) -> list[str]:
        if isinstance(self.model, list):
            return [m for m in self.model if m]
        return [self.model] if self.model else []


# A reference to a member: "name", "team:name" or an inline member definition.
MemberRef = Union[str, MemberConfig]


class RouteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    target: MemberRef


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    member: MemberRef
    instruction: str = ""


class TeamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["single", "mixture", "council", "critique", "plan", "vote", "route", "pipeline"]
    description: str = ""
    timeout: Optional[float] = Field(default=None, gt=0)
    prompts: dict[str, str] = Field(default_factory=dict)

    # single
    member: Optional[MemberRef] = None
    # mixture (mixture-of-agents)
    proposers: list[MemberRef] = Field(default_factory=list)
    aggregator: Optional[MemberRef] = None
    layers: int = Field(default=1, ge=1, le=4)
    min_success: int = Field(default=1, ge=1)
    # council (answer, anonymous peer review, chair)
    members: list[MemberRef] = Field(default_factory=list)
    reviewers: list[MemberRef] = Field(default_factory=list)
    chairman: Optional[MemberRef] = None
    # critique (draft, critique, revise)
    writer: Optional[MemberRef] = None
    critics: list[MemberRef] = Field(default_factory=list)
    max_rounds: int = Field(default=2, ge=1, le=6)
    approve_token: str = "APPROVED"
    # plan (planner, parallel workers, synthesizer)
    planner: Optional[MemberRef] = None
    workers: list[MemberRef] = Field(default_factory=list)
    synthesizer: Optional[MemberRef] = None
    max_subtasks: int = Field(default=5, ge=1, le=12)
    json_mode: bool = True
    # vote (self-consistency / judge)
    voters: list[MemberRef] = Field(default_factory=list)
    samples: int = Field(default=1, ge=1, le=8)
    judge: Optional[MemberRef] = None
    # route (classifier picks a specialist)
    router: Optional[MemberRef] = None
    routes: list[RouteConfig] = Field(default_factory=list)
    default: Optional[str] = None
    # pipeline (sequential stages)
    stages: list[StageConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _strategy_fields(self) -> "TeamConfig":
        s = self.strategy
        need: dict[str, list[str]] = {
            "single": ["member"],
            "mixture": ["proposers", "aggregator"],
            "council": ["members", "chairman"],
            "critique": ["writer", "critics"],
            "plan": ["planner", "workers", "synthesizer"],
            "vote": ["voters"],
            "route": ["router", "routes"],
            "pipeline": ["stages"],
        }
        missing = [f for f in need[s] if not getattr(self, f)]
        if missing:
            raise ValueError(f"strategy '{s}' requires: {', '.join(missing)}")
        if s == "route":
            names = [r.name for r in self.routes]
            if len(set(names)) != len(names):
                raise ValueError("route names must be unique")
            if self.default and self.default not in names:
                raise ValueError(f"default route '{self.default}' is not one of {names}")
        if s == "vote" and len(self.voters) * self.samples < 2:
            raise ValueError("vote needs at least two candidates (voters x samples)")
        return self


class StaticNodeConfig(BaseModel):
    """An inference server the coordinator calls directly, without an agent."""

    model_config = ConfigDict(extra="forbid")
    name: str
    url: str
    kind: Literal["ollama", "openai", "agent"] = "ollama"
    api_key: Optional[str] = None
    max_concurrency: int = Field(default=2, ge=1)
    labels: dict[str, str] = Field(default_factory=dict)
    # Optional fixed model list when the server cannot list its models.
    models: list[str] = Field(default_factory=list)


class ClusterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    heartbeat_timeout: float = Field(default=30, gt=0)
    node_expiry: float = Field(default=600, gt=0)
    request_timeout: float = Field(default=300, gt=0)
    connect_timeout: float = Field(default=5, gt=0)
    queue_timeout: float = Field(default=180, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    fallback_on_busy: bool = False
    failure_cooldown: float = Field(default=20, ge=0)
    probe_interval: float = Field(default=15, gt=0)
    static_nodes: list[StaticNodeConfig] = Field(default_factory=list)


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # List every model found on the nodes in /v1/models and allow calling them
    # directly (load-balanced across nodes), next to the teams.
    expose_models: bool = True
    # How team progress is shown to OpenAI-style streaming clients:
    # "reasoning" puts it in the reasoning field (Open WebUI shows it as a
    # collapsible thinking block); "none" sends only the final answer.
    progress: Literal["reasoning", "none"] = "reasoning"
    progress_field: str = "reasoning_content"
    team_prefix: str = "team/"
    trace_limit: int = Field(default=200, ge=1)
    max_depth: int = Field(default=4, ge=1, le=10)
    task_timeout: float = Field(default=900, gt=0)


class FlotillaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: ServerConfig = Field(default_factory=ServerConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    defaults: Params = Field(default_factory=Params)
    members: dict[str, MemberConfig] = Field(default_factory=dict)
    teams: dict[str, TeamConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-(.*?))?\}")


def interpolate_env(text: str, env: dict[str, str] | None = None) -> str:
    """Expand ${VAR} and ${VAR:-default} in the raw YAML text."""
    env = os.environ if env is None else env

    def repl(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        value = env.get(name)
        if value is None or value == "":
            return default if default is not None else ""
        return value

    return _ENV_REF.sub(repl, text)


def load_config(path: str | os.PathLike | None = None, text: str | None = None) -> FlotillaConfig:
    if text is None:
        if path is None:
            raise ConfigError("no configuration path given")
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        text = p.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(interpolate_env(text)) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("the config file must contain a mapping at the top level")
    try:
        cfg = FlotillaConfig.model_validate(raw)
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            problems.append(f"{loc}: {err['msg']}")
        raise ConfigError(problems) from exc
    problems = check_references(cfg)
    if problems:
        raise ConfigError(problems)
    return cfg


# ---------------------------------------------------------------------------
# reference checks and resolution
# ---------------------------------------------------------------------------


def team_member_refs(team: TeamConfig) -> list[tuple[str, MemberRef]]:
    """All (role, ref) pairs a team uses."""
    out: list[tuple[str, MemberRef]] = []

    def add(role: str, ref: MemberRef | None) -> None:
        if ref is not None:
            out.append((role, ref))

    add("member", team.member)
    for r in team.proposers:
        add("proposer", r)
    add("aggregator", team.aggregator)
    for r in team.members:
        add("council member", r)
    for r in team.reviewers:
        add("reviewer", r)
    add("chairman", team.chairman)
    add("writer", team.writer)
    for r in team.critics:
        add("critic", r)
    add("planner", team.planner)
    for r in team.workers:
        add("worker", r)
    add("synthesizer", team.synthesizer)
    for r in team.voters:
        add("voter", r)
    add("judge", team.judge)
    add("router", team.router)
    for route in team.routes:
        add(f"route '{route.name}'", route.target)
    for i, stage in enumerate(team.stages):
        add(f"stage {i + 1}", stage.member)
    return out


def _ref_team(cfg: FlotillaConfig, ref: MemberRef) -> str | None:
    """Name of the team a reference points to, if any."""
    if isinstance(ref, MemberConfig):
        return ref.team
    if ref.startswith("team:"):
        return ref[5:]
    member = cfg.members.get(ref)
    if member is not None and member.team:
        return member.team
    return None


def check_references(cfg: FlotillaConfig) -> list[str]:
    problems: list[str] = []
    for name, member in cfg.members.items():
        if member.team and member.team not in cfg.teams:
            problems.append(f"members.{name}: unknown team '{member.team}'")
    for tname, team in cfg.teams.items():
        for role, ref in team_member_refs(team):
            if isinstance(ref, MemberConfig):
                if ref.team and ref.team not in cfg.teams:
                    problems.append(f"teams.{tname} {role}: unknown team '{ref.team}'")
                continue
            if ref.startswith("team:"):
                if ref[5:] not in cfg.teams:
                    problems.append(f"teams.{tname} {role}: unknown team '{ref[5:]}'")
            elif ref not in cfg.members:
                problems.append(
                    f"teams.{tname} {role}: unknown member '{ref}' "
                    f"(define it under `members:` or use `team:<name>`)"
                )
    if problems:
        return problems
    # Detect cycles between teams (a team that eventually runs itself).
    graph = {
        t: {x for _, r in team_member_refs(team) if (x := _ref_team(cfg, r))}
        for t, team in cfg.teams.items()
    }
    state: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        state[node] = 1
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 1:
                cycle = path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt]
                problems.append("team cycle: " + " -> ".join(cycle))
            elif state.get(nxt) is None:
                visit(nxt, path + [nxt])
        state[node] = 2

    for t in sorted(graph):
        if state.get(t) is None:
            visit(t, [t])
    return problems


@dataclass
class ResolvedMember:
    """A member reference with defaults applied."""

    label: str                     # name shown in traces
    team: str | None = None        # set for nested teams
    candidates: list[str] = field(default_factory=list)
    params: Params = field(default_factory=Params)
    description: str = ""


_PARAM_FIELDS = tuple(Params.model_fields.keys())


def merge_params(defaults: Params, member: Params) -> Params:
    data: dict[str, Any] = {}
    for f in _PARAM_FIELDS:
        dv, mv = getattr(defaults, f), getattr(member, f)
        if f in ("labels", "prefer_labels", "extra"):
            merged = dict(dv or {})
            merged.update(mv or {})
            data[f] = merged
        else:
            data[f] = mv if mv is not None else dv
    return Params(**data)


def resolve_member(cfg: FlotillaConfig, ref: MemberRef, fallback_label: str = "inline") -> ResolvedMember:
    if isinstance(ref, MemberConfig):
        if ref.team:
            return ResolvedMember(label=f"team:{ref.team}", team=ref.team, description=ref.description)
        label = ref.candidates[0] if ref.candidates else fallback_label
        return ResolvedMember(
            label=label,
            candidates=ref.candidates,
            params=merge_params(cfg.defaults, ref),
            description=ref.description,
        )
    if ref.startswith("team:"):
        name = ref[5:]
        return ResolvedMember(label=ref, team=name, description=cfg.teams[name].description)
    member = cfg.members[ref]
    if member.team:
        return ResolvedMember(label=ref, team=member.team, description=member.description or cfg.teams[member.team].description)
    return ResolvedMember(
        label=ref,
        candidates=member.candidates,
        params=merge_params(cfg.defaults, member),
        description=member.description,
    )


def team_models(cfg: FlotillaConfig, team_name: str, seen: set[str] | None = None) -> set[str]:
    """Every model name a team may call, including nested teams."""
    seen = seen or set()
    if team_name in seen or team_name not in cfg.teams:
        return set()
    seen.add(team_name)
    models: set[str] = set()
    for _, ref in team_member_refs(cfg.teams[team_name]):
        res = resolve_member(cfg, ref)
        if res.team:
            models |= team_models(cfg, res.team, seen)
        else:
            models |= set(res.candidates)
    return models
