"""Collaboration strategies: how a team's members work together on one request.

Each strategy receives the client's chat messages and returns the final
answer. The step that produces the answer the user sees is marked `final`
and, when the client is streaming, streams straight to it through `sink`.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field

from flotilla.config import MemberRef, TeamConfig, resolve_member
from flotilla.engine import prompts as P
from flotilla.engine.context import RunContext, Sink, StepFailed, StepOutput
from flotilla.util import (
    extract_json,
    is_approval,
    letters,
    normalize_answer,
    parse_choice,
    parse_ranking,
    render_conversation,
    truncate,
    with_system,
)

CLIP = 4000  # characters of one answer/review passed on to the next model


def _clip(text: str | None) -> str:
    return truncate((text or "").strip(), CLIP)


def append_to_last_user(messages: list[dict], text: str) -> list[dict]:
    """Add text to the final user turn (keeps strict user/assistant alternation,
    which some chat templates require)."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[-1].get("role") == "user":
        content = msgs[-1].get("content")
        if isinstance(content, list):
            msgs[-1]["content"] = list(content) + [{"type": "text", "text": "\n\n" + text}]
        else:
            msgs[-1]["content"] = f"{content or ''}\n\n{text}"
    else:
        msgs.append({"role": "user", "content": text})
    return msgs


def _ok(results: list) -> list[StepOutput]:
    return [r for r in results if isinstance(r, StepOutput)]


def _why(results: list) -> str:
    """The first failure among parallel steps, to explain a team failure."""
    for r in results:
        if isinstance(r, StepFailed):
            return f": {r}"
    return ""


async def _emit_final(text: str, sink: Sink | None) -> None:
    """Send an answer that was produced without streaming (vote, approved draft)."""
    if sink and text:
        await sink(text)


# ---------------------------------------------------------------------------
# single
# ---------------------------------------------------------------------------


async def run_single(team: TeamConfig, messages: list[dict], ctx: RunContext, sink: Sink | None) -> str:
    out = await ctx.call(team.member, messages, role="member", final=True, sink=sink)
    return out.content


# ---------------------------------------------------------------------------
# mixture of agents
# ---------------------------------------------------------------------------


async def run_mixture(team: TeamConfig, messages: list[dict], ctx: RunContext, sink: Sink | None) -> str:
    answers: list[str] = []
    for layer in range(team.layers):
        if layer == 0:
            layer_msgs = messages
        else:
            block = P.labeled_answers([(str(i + 1), _clip(a)) for i, a in enumerate(answers)])
            layer_msgs = with_system(messages, P.render(P.get(team.prompts, "refine"), answers=block))
        suffix = f" (layer {layer + 1})" if team.layers > 1 else ""
        results = await ctx.gather([
            ctx.call(ref, layer_msgs, role="proposer", label=f"proposer {i + 1}{suffix}")
            for i, ref in enumerate(team.proposers)
        ])
        ok = _ok(results)
        if len(ok) < team.min_success:
            raise StepFailed(
                f"only {len(ok)} of {len(team.proposers)} proposers answered (need {team.min_success}){_why(results)}"
            )
        answers = [r.content for r in ok]
    block = P.labeled_answers([(str(i + 1), _clip(a)) for i, a in enumerate(answers)])
    agg_msgs = with_system(messages, P.render(P.get(team.prompts, "aggregate"), answers=block))
    out = await ctx.call(team.aggregator, agg_msgs, role="aggregator", final=True, sink=sink)
    return out.content


# ---------------------------------------------------------------------------
# council: answer -> anonymous peer review -> chair
# ---------------------------------------------------------------------------


async def run_council(team: TeamConfig, messages: list[dict], ctx: RunContext, sink: Sink | None) -> str:
    task_text = render_conversation(messages)
    results = await ctx.gather([
        ctx.call(ref, messages, role="council member", label=f"member {i + 1}")
        for i, ref in enumerate(team.members)
    ])
    ok = _ok(results)
    if not ok:
        raise StepFailed(f"no council member answered{_why(results)}")
    labs = letters(len(ok))
    answers_block = P.labeled_answers([(lab, _clip(r.content)) for lab, r in zip(labs, ok)], noun="Response")
    reviews_block = "(no reviews)"
    ranking = ", ".join(labs)
    if len(ok) >= 2:
        prompt = P.render(
            P.get(team.prompts, "review"),
            task=task_text, answers=answers_block, example_ranking=", ".join(reversed(labs)),
        )
        reviewers = team.reviewers or team.members
        reviews = _ok(await ctx.gather([
            ctx.call(ref, [{"role": "user", "content": prompt}], role="reviewer", label=f"reviewer {i + 1}")
            for i, ref in enumerate(reviewers)
        ]))
        if reviews:
            scores = {lab: 0 for lab in labs}
            for r in reviews:
                for pos, lab in enumerate(parse_ranking(r.content, labs)):
                    scores[lab] += len(labs) - pos
            ranked = sorted(labs, key=lambda lab: (-scores[lab], labs.index(lab)))
            ranking = ", ".join(f"{lab} ({scores[lab]} points)" for lab in ranked)
            reviews_block = "\n\n".join(f"[Review {i + 1}]\n{_clip(r.content)}" for i, r in enumerate(reviews))
            ctx.note(f"peer ranking: {ranking}")
    chair_msgs = with_system(messages, P.render(
        P.get(team.prompts, "chair"), answers=answers_block, reviews=reviews_block, ranking=ranking,
    ))
    out = await ctx.call(team.chairman, chair_msgs, role="chairman", final=True, sink=sink)
    return out.content


# ---------------------------------------------------------------------------
# critique: draft -> critics -> revise (repeat)
# ---------------------------------------------------------------------------


async def run_critique(team: TeamConfig, messages: list[dict], ctx: RunContext, sink: Sink | None) -> str:
    task_text = render_conversation(messages)
    draft = (await ctx.call(team.writer, messages, role="writer", label="draft")).content
    for rnd in range(1, team.max_rounds + 1):
        prompt = P.render(P.get(team.prompts, "critic"), task=task_text, draft=_clip(draft), approve_token=team.approve_token)
        reviews = _ok(await ctx.gather([
            ctx.call(ref, [{"role": "user", "content": prompt}], role="critic", label=f"critic {i + 1}, round {rnd}")
            for i, ref in enumerate(team.critics)
        ]))
        if not reviews:
            ctx.note("no critic answered; keeping the current draft")
            break
        issues = [r.content for r in reviews if not is_approval(r.content, team.approve_token)]
        if not issues:
            ctx.note(f"round {rnd}: every critic approved the draft")
            break
        ctx.note(f"round {rnd}: {len(issues)} of {len(reviews)} critics asked for changes")
        feedback = "\n\n".join(f"[Reviewer {i + 1}]\n{_clip(t)}" for i, t in enumerate(issues))
        revise_msgs = list(messages) + [
            {"role": "assistant", "content": draft},
            {"role": "user", "content": P.render(P.get(team.prompts, "revise"), feedback=feedback)},
        ]
        last = rnd == team.max_rounds
        out = await ctx.call(
            team.writer, revise_msgs, role="writer", label=f"revision {rnd}",
            final=last, sink=sink if last else None,
        )
        draft = out.content
        if last:
            return draft
    await _emit_final(draft, sink)
    return draft


# ---------------------------------------------------------------------------
# plan: planner -> parallel workers (with dependencies) -> synthesizer
# ---------------------------------------------------------------------------


@dataclass
class Subtask:
    id: str
    title: str
    instructions: str
    worker: str
    depends_on: list[str] = field(default_factory=list)


def worker_names(team: TeamConfig, ctx: RunContext) -> list[tuple[str, MemberRef, str]]:
    """(unique name, ref, description) for each worker."""
    out: list[tuple[str, MemberRef, str]] = []
    used: Counter = Counter()
    for ref in team.workers:
        res = resolve_member(ctx.cfg, ref)
        base = res.label
        used[base] += 1
        name = base if used[base] == 1 else f"{base}#{used[base]}"
        out.append((name, ref, res.description))
    return out


def parse_plan(text: str, names: list[str], max_subtasks: int) -> list[Subtask] | None:
    data = extract_json(text)
    if isinstance(data, dict):
        items = data.get("subtasks") or data.get("tasks") or data.get("steps") or data.get("plan")
    else:
        items = data
    if not isinstance(items, list) or not items:
        return None
    lowered = {n.lower(): n for n in names}
    subtasks: list[Subtask] = []
    seen: set[str] = set()
    for i, item in enumerate(items[:max_subtasks]):
        if isinstance(item, str):
            item = {"title": truncate(item, 80), "instructions": item}
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or f"s{i + 1}").strip() or f"s{i + 1}"
        while sid in seen:
            sid += "x"
        title = str(item.get("title") or item.get("name") or f"Subtask {i + 1}").strip()
        instructions = str(item.get("instructions") or item.get("description") or item.get("task") or title).strip()
        wanted = str(item.get("worker") or item.get("assignee") or "").strip().lower()
        worker = lowered.get(wanted) or names[i % len(names)]
        deps_raw = item.get("depends_on") or item.get("dependencies") or []
        if isinstance(deps_raw, (str, int)):
            deps_raw = [deps_raw]
        deps = [str(d) for d in deps_raw if str(d) in seen]   # only earlier ids: no cycles
        subtasks.append(Subtask(sid, truncate(title, 120), instructions, worker, deps))
        seen.add(sid)
    return subtasks or None


async def run_plan(team: TeamConfig, messages: list[dict], ctx: RunContext, sink: Sink | None) -> str:
    task_text = render_conversation(messages)
    workers = worker_names(team, ctx)
    names = [w[0] for w in workers]
    refs = {w[0]: w[1] for w in workers}
    roster = "\n".join(f"- {name}: {desc or 'general-purpose assistant'}" for name, _, desc in workers)
    plan_msgs = [{"role": "user", "content": P.render(
        P.get(team.prompts, "plan"), max_subtasks=team.max_subtasks, workers=roster, task=task_text,
    )}]
    response_format = {"type": "json_object"} if team.json_mode else None
    subtasks: list[Subtask] | None = None
    for attempt in range(2):
        try:
            out = await ctx.call(
                team.planner, plan_msgs, role="planner",
                label="plan" if attempt == 0 else "plan (retry)", response_format=response_format,
            )
        except StepFailed:
            if attempt == 0:
                response_format = None  # the server may not support JSON mode
                continue
            break
        subtasks = parse_plan(out.content, names, team.max_subtasks)
        if subtasks:
            break
        plan_msgs = plan_msgs + [
            {"role": "assistant", "content": out.content},
            {"role": "user", "content": "That was not valid JSON in the required shape. Reply with only the JSON object."},
        ]
    if not subtasks:
        ctx.note("the plan was unusable; handling the request as one subtask")
        subtasks = [Subtask("s1", "Complete the request", "Answer the request completely.", names[0])]
    ctx.note("plan: " + "; ".join(
        f"{s.id} {s.title} -> {s.worker}" + (f" (after {', '.join(s.depends_on)})" if s.depends_on else "")
        for s in subtasks
    ))

    results: dict[str, str | None] = {}
    titles = {s.id: s.title for s in subtasks}
    running: dict[str, asyncio.Task] = {}

    async def work(st: Subtask) -> None:
        for dep in st.depends_on:
            await running[dep]
        context = ""
        if st.depends_on:
            context = "\n\nResults of the subtasks this one builds on:\n\n" + "\n\n".join(
                f"[{d}: {titles[d]}]\n{_clip(results.get(d)) or '(that subtask failed)'}" for d in st.depends_on
            )
        user = P.render(P.get(team.prompts, "work_input"), task=task_text, title=st.title,
                        instructions=st.instructions, context=context)
        msgs = [{"role": "system", "content": P.get(team.prompts, "work")}, {"role": "user", "content": user}]
        try:
            out = await ctx.call(refs[st.worker], msgs, role="worker", label=f"{st.id}: {st.title}")
            results[st.id] = out.content
        except StepFailed:
            results[st.id] = None

    for st in subtasks:
        running[st.id] = asyncio.ensure_future(work(st))
    try:
        await asyncio.gather(*running.values())
    finally:
        for t in running.values():
            if not t.done():
                t.cancel()
    if not any(results.values()):
        raise StepFailed("every subtask failed")
    blocks = "\n\n".join(
        f"[Subtask {s.id}: {s.title}]\n{_clip(results.get(s.id)) or '(this subtask failed)'}" for s in subtasks
    )
    syn_msgs = with_system(messages, P.render(P.get(team.prompts, "synthesize"), results=blocks))
    out = await ctx.call(team.synthesizer, syn_msgs, role="synthesizer", final=True, sink=sink)
    return out.content


# ---------------------------------------------------------------------------
# vote: several answers, then a judge or a majority
# ---------------------------------------------------------------------------


def majority(texts: list[str]) -> tuple[str, int]:
    keys = [normalize_answer(t) for t in texts]
    counts = Counter(keys)
    best_key = max(keys, key=lambda k: (counts[k], -keys.index(k)))
    return texts[keys.index(best_key)], counts[best_key]


async def run_vote(team: TeamConfig, messages: list[dict], ctx: RunContext, sink: Sink | None) -> str:
    calls = []
    for i, ref in enumerate(team.voters):
        for s in range(team.samples):
            label = f"voter {i + 1}" + (f", sample {s + 1}" if team.samples > 1 else "")
            calls.append(ctx.call(ref, messages, role="voter", label=label))
    results = await ctx.gather(calls)
    ok = _ok(results)
    if not ok:
        raise StepFailed(f"no voter answered{_why(results)}")
    texts = [r.content for r in ok]
    if len(texts) == 1:
        winner = texts[0]
    elif team.judge:
        labs = letters(len(texts))
        prompt = P.render(
            P.get(team.prompts, "judge"),
            task=render_conversation(messages),
            answers=P.labeled_answers([(lab, _clip(t)) for lab, t in zip(labs, texts)], noun="Candidate"),
        )
        choice = None
        try:
            verdict = await ctx.call(team.judge, [{"role": "user", "content": prompt}], role="judge")
            choice = parse_choice(verdict.content, labs, "BEST")
        except StepFailed:
            pass
        if choice:
            winner = texts[labs.index(choice)]
            ctx.note(f"judge picked candidate {choice}")
        else:
            winner, count = majority(texts)
            ctx.note(f"judge verdict unreadable; majority answer ({count} of {len(texts)})")
    else:
        winner, count = majority(texts)
        ctx.note(f"majority answer: {count} of {len(texts)} agree")
    await _emit_final(winner, sink)
    return winner


# ---------------------------------------------------------------------------
# route: a small model picks the specialist
# ---------------------------------------------------------------------------


def parse_route(text: str, names: list[str]) -> str | None:
    lowered = {n.lower(): n for n in names}
    data = extract_json(text)
    if isinstance(data, dict):
        for key in ("route", "name", "choice", "category"):
            val = data.get(key)
            if isinstance(val, str) and val.strip().lower() in lowered:
                return lowered[val.strip().lower()]
    body = (text or "").lower()
    positions = []
    for low, name in lowered.items():
        m = re.search(rf"(?<![\w-]){re.escape(low)}(?![\w-])", body)
        if m:
            positions.append((m.start(), name))
    return min(positions)[1] if positions else None


async def run_route(team: TeamConfig, messages: list[dict], ctx: RunContext, sink: Sink | None) -> str:
    names = [r.name for r in team.routes]
    routes_desc = "\n".join(f"- {r.name}: {r.description or 'no description'}" for r in team.routes)
    prompt = P.render(P.get(team.prompts, "route"), routes=routes_desc, task=render_conversation(messages, limit=6000))
    chosen = None
    try:
        out = await ctx.call(
            team.router, [{"role": "user", "content": prompt}], role="router",
            response_format={"type": "json_object"} if team.json_mode else None,
        )
        chosen = parse_route(out.content, names)
    except StepFailed as exc:
        ctx.note(f"router failed ({exc}); using the default route")
    if chosen is None:
        chosen = team.default or names[0]
        ctx.note(f"no clear route; using '{chosen}'")
    else:
        ctx.note(f"route: {chosen}")
    route = next(r for r in team.routes if r.name == chosen)
    out = await ctx.call(route.target, messages, role="route", label=f"route: {chosen}", final=True, sink=sink)
    return out.content


# ---------------------------------------------------------------------------
# pipeline: fixed sequence of stages
# ---------------------------------------------------------------------------


async def run_pipeline(team: TeamConfig, messages: list[dict], ctx: RunContext, sink: Sink | None) -> str:
    previous = ""
    last_index = len(team.stages) - 1
    for i, stage in enumerate(team.stages):
        if i == 0:
            msgs = with_system(messages, stage.instruction) if stage.instruction else messages
        else:
            instruction = stage.instruction or "Improve it. Reply with only the improved version."
            msgs = append_to_last_user(messages, P.render(
                P.get(team.prompts, "stage_input"), previous=_clip(previous), instruction=instruction,
            ))
        last = i == last_index
        out = await ctx.call(
            stage.member, msgs, role="stage", label=f"stage {i + 1}",
            final=last, sink=sink if last else None,
        )
        previous = out.content
    return previous


STRATEGIES = {
    "single": run_single,
    "mixture": run_mixture,
    "council": run_council,
    "critique": run_critique,
    "plan": run_plan,
    "vote": run_vote,
    "route": run_route,
    "pipeline": run_pipeline,
}
