"""Command line: `python -m flotilla <command>` (or `flotilla <command>` when installed)."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import textwrap

import httpx

from flotilla import __version__


def _client_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", default=os.environ.get("FLOTILLA_URL", "http://127.0.0.1:8800"),
                   help="coordinator URL (env FLOTILLA_URL)")
    # Inside the coordinator container the first configured key works too.
    default_key = os.environ.get("FLOTILLA_API_KEY") or next(
        (k.strip() for k in os.environ.get("FLOTILLA_API_KEYS", "").replace(";", ",").split(",") if k.strip()), None
    )
    p.add_argument("--key", default=default_key, help="API key (env FLOTILLA_API_KEY)")


def _http(args) -> httpx.Client:
    headers = {"authorization": f"Bearer {args.key}"} if args.key else {}
    return httpx.Client(base_url=args.url.rstrip("/"), headers=headers, timeout=httpx.Timeout(900, connect=10), trust_env=False)


def _die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _get(args, path: str) -> dict:
    try:
        with _http(args) as c:
            r = c.get(path)
    except httpx.HTTPError as exc:
        _die(f"cannot reach coordinator at {args.url}: {exc}")
    if r.status_code >= 400:
        _die(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def cmd_coordinator(args) -> None:
    from flotilla.coordinator import serve
    from flotilla.settings import CoordinatorSettings

    s = CoordinatorSettings.from_env()
    if args.config:
        s.config_path = args.config
    if args.port:
        s.port = args.port
    if args.host:
        s.host = args.host
    serve(s)


def cmd_agent(args) -> None:
    from flotilla.agent import serve
    from flotilla.settings import AgentSettings

    s = AgentSettings.from_env()
    if args.port:
        s.port = args.port
    serve(s)


def cmd_fake_backend(args) -> None:
    from flotilla.fake_backend import serve

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    serve(args.name, models, args.host, args.port, args.latency, args.mode)


def cmd_check_config(args) -> None:
    from flotilla.config import ConfigError, load_config, team_models

    path = args.path or os.environ.get("FLOTILLA_CONFIG", "config/flotilla.yaml")
    try:
        cfg = load_config(path)
    except ConfigError as exc:
        print(f"{path}: invalid", file=sys.stderr)
        for p in exc.problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(2)
    print(f"{path}: OK")
    print(f"  members: {', '.join(cfg.members) or 'none'}")
    for name, team in cfg.teams.items():
        print(f"  team {name} ({team.strategy}): models {', '.join(sorted(team_models(cfg, name)))}")
    all_models = sorted({m for t in cfg.teams for m in team_models(cfg, t)})
    print(f"  models referenced: {', '.join(all_models)}")


def cmd_nodes(args) -> None:
    data = _get(args, "/api/cluster")
    if args.json:
        print(json.dumps(data, indent=2))
        return
    t = data["totals"]
    print(f"cluster {data['cluster']}: {t['online']}/{t['nodes']} nodes online, {t['in_flight']}/{t['slots']} slots busy")
    for n in data["nodes"]:
        models = ", ".join(m["name"] for m in n["models"]) or "-"
        print(f"  {n['name']:<18} {n['status']:<11} {n['in_flight']}/{n['capacity']:<3} {n['url']}")
        print(f"  {'':<18} models: {models}")
        if n.get("last_error") and n["status"] != "online":
            print(f"  {'':<18} last error: {n['last_error'][:160]}")


def cmd_teams(args) -> None:
    data = _get(args, "/api/teams")
    for t in data["teams"]:
        state = "ready" if t["ready"] else f"missing models: {', '.join(t['missing_models'])}"
        print(f"  {t['id']:<22} {t['strategy']:<9} {state}")
        if t["description"]:
            print(f"  {'':<22} {t['description']}")


def cmd_run(args) -> None:
    prompt = args.prompt if args.prompt != "-" else sys.stdin.read()
    body = {"messages": [{"role": "user", "content": prompt}], "source": "cli"}
    if args.system:
        body["messages"].insert(0, {"role": "system", "content": args.system})
    if args.model:
        body["model"] = args.model
    else:
        body["team"] = args.team
    try:
        with _http(args) as c:
            r = c.post("/api/tasks", json=body)
    except httpx.HTTPError as exc:
        _die(f"cannot reach coordinator at {args.url}: {exc}")
    if r.status_code >= 400:
        _die(f"HTTP {r.status_code}: {r.text[:500]}")
    task = r.json()
    if args.json:
        print(json.dumps(task, indent=2))
        return
    if args.trace:
        print(f"task {task['id']}  {task['status']}  {task['seconds']}s  "
              f"tokens {task['usage']['total_tokens']}  nodes {', '.join(task['nodes']) or '-'}")
        for s in task["step_details"]:
            indent = "    " if s.get("parent") else "  "
            where = f"{s.get('model') or s.get('member')} @ {s.get('node') or '-'}"
            print(f"{indent}{s['status']:<5} {s['label'][:40]:<40} {where:<40} {s.get('seconds') or ''}s")
            if s.get("error"):
                print(f"{indent}      {s['error'][:200]}")
        print()
    if task["status"] != "ok":
        _die(f"failed: {task.get('error')}")
    print(task["output"])


def cmd_pull(args) -> None:
    body = {"model": args.model}
    if args.node:
        body["nodes"] = args.node
    with _http(args) as c:
        r = c.post("/api/pull", json=body)
    if r.status_code >= 400:
        _die(f"HTTP {r.status_code}: {r.text[:500]}")
    for node, res in r.json()["nodes"].items():
        print(f"  {node}: {res.get('status') or ('ok' if res.get('ok') else res.get('error'))}")


def cmd_reload(args) -> None:
    with _http(args) as c:
        r = c.post("/api/config/reload")
    data = r.json()
    if not data.get("ok"):
        print("reload failed:", file=sys.stderr)
        for p in data.get("problems", []):
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(2)
    print(f"reloaded: teams {', '.join(data['teams'])}")


def cmd_healthcheck(args) -> None:
    scheme = "https" if os.environ.get("FLOTILLA_TLS_CERT") else "http"
    url = args.url or f"{scheme}://127.0.0.1:{args.port or os.environ.get('FLOTILLA_PORT', '8800')}/health"
    try:
        r = httpx.get(url, timeout=5, trust_env=False, verify=False)
        raise SystemExit(0 if r.status_code == 200 else 1)
    except httpx.HTTPError:
        raise SystemExit(1)


def cmd_token(args) -> None:
    print(secrets.token_urlsafe(32))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="flotilla",
        description="Run many small LLMs across machines as one team.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              flotilla coordinator --config config/flotilla.yaml
              flotilla agent                    (configured with FLOTILLA_* variables)
              flotilla run moa "Explain RAID levels" --trace
              flotilla nodes
        """),
    )
    parser.add_argument("--version", action="version", version=f"flotilla {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("coordinator", help="run the coordinator (API, scheduler, dashboard)")
    p.add_argument("--config")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_coordinator)

    p = sub.add_parser("agent", help="run a node agent next to Ollama/vLLM/llama.cpp")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("fake-backend", help="run a fake inference server for testing")
    p.add_argument("--name", default="fake")
    p.add_argument("--models", default="qwen3.5:4b,granite4.2:3b")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--latency", type=float, default=0.05)
    p.add_argument("--mode", choices=["ollama", "openai"], default="ollama")
    p.set_defaults(func=cmd_fake_backend)

    p = sub.add_parser("check-config", help="validate a team configuration file")
    p.add_argument("path", nargs="?")
    p.set_defaults(func=cmd_check_config)

    p = sub.add_parser("nodes", help="list nodes and their models")
    _client_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_nodes)

    p = sub.add_parser("teams", help="list teams and whether their models are available")
    _client_args(p)
    p.set_defaults(func=cmd_teams)

    p = sub.add_parser("run", help="run a team (or a single model) on a prompt")
    _client_args(p)
    p.add_argument("team", nargs="?", default="moa", help="team name (default: moa)")
    p.add_argument("prompt", help="the request, or - to read stdin")
    p.add_argument("--model", help="call one model directly instead of a team")
    p.add_argument("--system", help="system prompt")
    p.add_argument("--trace", action="store_true", help="print every step (model, node, time)")
    p.add_argument("--json", action="store_true", help="print the full task record")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pull", help="download a model on every agent node (Ollama)")
    _client_args(p)
    p.add_argument("model")
    p.add_argument("--node", action="append", help="only this node (repeatable)")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("reload", help="reload the coordinator's team configuration")
    _client_args(p)
    p.set_defaults(func=cmd_reload)

    p = sub.add_parser("healthcheck", help="exit 0 if the local service is healthy (for Docker)")
    p.add_argument("--port", type=int)
    p.add_argument("--url")
    p.set_defaults(func=cmd_healthcheck)

    p = sub.add_parser("token", help="print a random token for FLOTILLA_CLUSTER_TOKEN / API keys")
    p.set_defaults(func=cmd_token)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
