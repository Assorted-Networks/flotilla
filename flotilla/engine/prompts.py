"""Prompt templates for the collaboration strategies.

Every template can be overridden per team with `prompts: {key: "..."}` in the
config. Placeholders use {{double_braces}} so JSON examples stay literal.
"""

from __future__ import annotations

import re

DEFAULTS: dict[str, str] = {
    # -- mixture -------------------------------------------------------------
    "aggregate": (
        "You are the final writer on a team of AI models. Several models answered the "
        "user's latest request independently; their answers are below. Some of them may "
        "contain mistakes, omissions or unsupported claims.\n\n"
        "Write the single best response to the user: keep what is correct and useful, fix "
        "or drop what is wrong, settle disagreements by reasoning about which answer is "
        "right, and add anything important they all missed. Do not mention the other "
        "models or that you are combining answers; reply directly to the user.\n\n"
        "{{answers}}"
    ),
    "refine": (
        "Other AI models answered the same request; their answers are below and may "
        "contain mistakes. Use them as extra input, then write your own improved, "
        "self-contained answer to the user's latest request.\n\n{{answers}}"
    ),
    # -- council -------------------------------------------------------------
    "review": (
        "You are reviewing anonymous answers to the user's request below.\n\n"
        "Request:\n{{task}}\n\n{{answers}}\n\n"
        "For each answer, briefly note its strengths and any concrete errors or gaps. "
        "Then rank all answers from best to worst. End your reply with one line in "
        "exactly this format, listing every label once:\n"
        "FINAL RANKING: {{example_ranking}}"
    ),
    "chair": (
        "You chair a council of AI models. The council members answered the user's "
        "latest request, then reviewed each other's answers anonymously. The answers, "
        "the reviews and the combined ranking are below.\n\n"
        "Write the final response to the user: build on the strongest answer, include "
        "valid points from the others, and correct every error the reviewers found. Do "
        "not mention the council, the reviews or the ranking.\n\n"
        "{{answers}}\n\n{{reviews}}\n\nCombined ranking (best first): {{ranking}}"
    ),
    # -- critique ------------------------------------------------------------
    "critic": (
        "You are a strict reviewer. Check the draft below against the user's request.\n\n"
        "Request:\n{{task}}\n\nDraft:\n{{draft}}\n\n"
        "If the draft fully and correctly satisfies the request, reply with exactly: "
        "{{approve_token}}\n"
        "Otherwise list the specific problems (errors, missing parts, unclear passages) "
        "as short bullet points, most important first. Do not rewrite the draft."
    ),
    "revise": (
        "Reviewers checked your draft and found the problems below. Revise the draft to "
        "fix them. Reply with only the complete revised response for the user, with no "
        "preamble and no notes about what changed.\n\n{{feedback}}"
    ),
    # -- plan ----------------------------------------------------------------
    "plan": (
        "You are the planner for a team of AI workers. Split the user's request into at "
        "most {{max_subtasks}} subtasks that workers can complete separately; a writer "
        "will combine their results afterwards. Use fewer subtasks for simple requests; "
        "one is fine. Each subtask must be self-contained, because workers only see "
        "their own instructions and the original request.\n\n"
        "Available workers:\n{{workers}}\n\n"
        "Request:\n{{task}}\n\n"
        "Reply with only JSON in this shape:\n"
        '{"subtasks": [{"id": "s1", "title": "short title", "instructions": "what to do", '
        '"worker": "<worker name>", "depends_on": []}]}\n'
        "Use depends_on (a list of earlier ids) only when a subtask needs another "
        "subtask's result."
    ),
    "work": (
        "You are one worker on a team handling a larger request. Complete only your "
        "assigned subtask, thoroughly but without padding. A writer will merge your "
        "result with the other workers' results."
    ),
    "work_input": (
        "Overall request:\n{{task}}\n\nYour subtask: {{title}}\n{{instructions}}{{context}}"
    ),
    "synthesize": (
        "You are the final writer for a team of AI workers. The user's request was split "
        "into subtasks and the workers' results are below. Write the complete final "
        "response to the user's latest request from these results: merge them into one "
        "coherent answer, remove repetition, resolve inconsistencies and fill small gaps. "
        "Do not mention subtasks, workers or the team.\n\n{{results}}"
    ),
    # -- vote ----------------------------------------------------------------
    "judge": (
        "Several candidate answers to the user's request are below, labeled with letters.\n\n"
        "Request:\n{{task}}\n\n{{answers}}\n\n"
        "Decide which single answer is best: correct, complete and directly responsive. "
        "Explain your choice in one or two sentences, then end with one line in exactly "
        "this format:\nBEST: <label>"
    ),
    # -- route ---------------------------------------------------------------
    "route": (
        "Pick the best route for the user's request.\n\nRoutes:\n{{routes}}\n\n"
        "Request:\n{{task}}\n\n"
        'Reply with only JSON: {"route": "<route name>"}'
    ),
    # -- pipeline ------------------------------------------------------------
    "stage_input": (
        "Here is the current version from the previous step:\n\n{{previous}}\n\n{{instruction}}"
    ),
}

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def render(template: str, **values: object) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        return str(values.get(key, ""))

    return _PLACEHOLDER.sub(repl, template)


def get(overrides: dict[str, str], key: str) -> str:
    return overrides.get(key) or DEFAULTS[key]


def labeled_answers(answers: list[tuple[str, str]], noun: str = "Answer") -> str:
    """Format [(label, text)] as blocks such as `[Answer A]\\n...`."""
    return "\n\n".join(f"[{noun} {label}]\n{text.strip()}" for label, text in answers)
