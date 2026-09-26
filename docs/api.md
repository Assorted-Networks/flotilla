# HTTP API

All endpoints except `/health` and `/` need an API key when `FLOTILLA_API_KEYS`
is set: `Authorization: Bearer <key>` or `x-api-key: <key>`. Errors use the
OpenAI shape: `{"error": {"message", "type", "code"}}`.

## OpenAI-compatible

### `GET /v1/models`
Teams (`team/<name>`) and, with `server.expose_models`, every model on an
online machine (`owned_by` lists the machines).

### `POST /v1/chat/completions`
Standard chat completion request.

- `model: "team/<name>"` runs a team. `stream`, `stream_options.include_usage`
  and `max_tokens` (applied to the step that writes the final answer) are
  honoured; other sampling fields, `tools` and `n` are ignored for teams.
  Progress lines arrive in `reasoning_content` deltas while the team works;
  the final answer arrives as normal `content` deltas. Non-streaming responses
  carry the progress log in `message.reasoning_content` and a `flotilla`
  object: `{"task_id", "steps", "nodes"}`.
- `model: "<model name>"` sends the request, unchanged, to a machine that has
  that model (least loaded first), with failover until the first token. Tools,
  images and other fields pass through.

Every chat response has an `x-flotilla-task-id` header. If a streamed team run
fails after the response has started, the stream ends with an
`{"error": {...}}` event and `[DONE]`.

### `POST /v1/embeddings`
Passed through to a machine that has the embedding model.

## Native

### `POST /api/tasks`
Run a team or a single model and record a trace.

```json
{"team": "council", "prompt": "Is RAID a backup?", "system": "optional", "wait": true}
{"model": "qwen3.5:4b", "messages": [{"role": "user", "content": "hi"}], "wait": false}
```

`wait: true` (default) returns the finished task. `wait: false` returns
`202 {"id", "status", "events"}` immediately; follow it with the events stream.

A task:

```json
{
  "id": "task_...", "target": "council", "kind": "team", "status": "ok",
  "seconds": 12.4, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "nodes": ["node-a", "node-b"], "models": ["granite4.2:3b", "qwen3.5:9b"],
  "output": "...", "error": null, "messages": [...],
  "step_details": [
    {"id": "step_...", "parent": null, "role": "council member", "label": "member 1",
     "member": "qwen", "kind": "llm", "status": "ok", "model": "qwen3.5:4b", "node": "node-a",
     "started_at": 0, "assigned_at": 0, "ended_at": 0, "seconds": 3.1,
     "usage": {...}, "output": "...", "reasoning": null, "attempts": [], "final": false}
  ]
}
```

`status` is `running`, `ok`, `error` or `cancelled` (the client disconnected).
Steps of nested teams have `parent` set to the enclosing team step.
`attempts` lists failed tries on other machines before the step succeeded.

### `GET /api/tasks?limit=50`
Recent tasks, newest first (summaries).

### `GET /api/tasks/{id}`
One task with all steps.

### `GET /api/tasks/{id}/events`
Server-sent events. Past events are replayed first, then live ones until the
task ends. Event types:

| Type | Fields |
| --- | --- |
| `task.started` | `target`, `kind`, `strategy` |
| `team.started` / `team.completed` | `team`, `strategy`, `parent`, `depth` |
| `step.started` | `step` (summary) |
| `step.assigned` | `step` (id), `label`, `node`, `model` (sent again on each retry) |
| `step.retry` | `step`, `reason` |
| `step.delta` | `step`, `reasoning` (thinking text of the final step) |
| `step.completed` / `step.failed` | `step` |
| `task.note` | `text` (plan, ranking, route decisions) |
| `task.delta` | `content` (final answer as it streams) |
| `task.completed` / `task.failed` | `status`, `output`, `error`, `usage`, `seconds` |

### `GET /api/cluster`
Machines (`status`: `online`, `cooldown`, `unreachable`, `offline`; models,
loaded models, slots, hardware, labels, request counts, speed, last error),
the model index, and totals.

### `GET /api/teams`
Each team's strategy, roles, candidate models, which models are available
right now, `missing_models` and `ready`.

### `POST /api/pull`
`{"model": "qwen3.5:9b", "nodes": ["gpu-box"]}` asks agents with an Ollama
backend to download a model (all agents when `nodes` is omitted). Returns per
machine whether the download started; progress is in each agent's log and
`GET <agent>/admin/pull`.

### `POST /api/config/reload`
Re-read the team file. `400 {"ok": false, "problems": [...]}` when it is
invalid (the old configuration stays active).

### `GET /metrics`
Prometheus text format.

### `GET /health`
`{"status": "ok", "version", "nodes_online"}`, no key needed.

## Agent endpoints

Used by the coordinator; all but `/health` need the cluster token.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness (used for the reachability check) |
| `GET /info` | Models, loaded models, load, hardware, downloads |
| `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/embeddings` | Proxied to the local model server |
| `GET/POST /admin/pull` | Download status / start a download (Ollama) |

The coordinator's own agent endpoints are `POST /api/agents/heartbeat` and
`POST /api/agents/deregister` (cluster token).
