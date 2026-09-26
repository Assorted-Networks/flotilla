# Configuration

Flotilla has two kinds of settings:

- **The team file** (`config/flotilla.yaml`, path in `FLOTILLA_CONFIG`): members,
  teams, scheduling. Read by the coordinator; reload it without a restart with
  `flotilla reload` or `POST /api/config/reload`. Validate it with
  `flotilla check-config path/to/file.yaml`.
- **Environment variables**: secrets and wiring for the coordinator and each agent.

`${VAR}` and `${VAR:-default}` inside the YAML are replaced from the
coordinator's environment before parsing. Unknown keys are rejected, so typos
fail loudly instead of being ignored.

## Team file

```yaml
server:      {...}   # API behaviour
cluster:     {...}   # scheduling, timeouts, static nodes
defaults:    {...}   # generation settings applied to every member
members:     {...}   # named models with a role
teams:       {...}   # members + a strategy
```

### `server`

| Key | Default | Meaning |
| --- | --- | --- |
| `expose_models` | `true` | List every model found on the machines in `/v1/models` and allow calling them directly (load-balanced, with failover). |
| `progress` | `reasoning` | `reasoning` streams team progress in the reasoning field of each chunk (Open WebUI shows it as a thinking block); `none` sends only the answer. |
| `progress_field` | `reasoning_content` | Field name used for progress (`reasoning` also works with most clients). |
| `team_prefix` | `team/` | Teams are listed as `<prefix><name>`. A request for a bare team name also works when no model has that name. |
| `trace_limit` | `200` | Finished tasks kept in memory (and in `$FLOTILLA_DATA_DIR/tasks.jsonl`). |
| `max_depth` | `4` | Maximum nesting of teams inside teams. |
| `task_timeout` | `900` | Seconds a whole team run may take (per-team `timeout` overrides it). |

### `cluster`

| Key | Default | Meaning |
| --- | --- | --- |
| `heartbeat_timeout` | `30` | Seconds without a heartbeat before an agent counts as offline. Agents send one every 10 s. |
| `node_expiry` | `600` | Offline agents are forgotten after this long. |
| `request_timeout` | `300` | Upper bound for one model call (seconds between received bytes). |
| `connect_timeout` | `5` | Connecting to a machine. |
| `queue_timeout` | `180` | How long a call may wait for a free slot before failing. |
| `max_retries` | `2` | Extra attempts on other machines after a failure. |
| `fallback_on_busy` | `false` | `false`: when a member's first-choice model exists but every machine with it is busy, wait for it. `true`: use the next model in the member's list instead. |
| `failure_cooldown` | `20` | Seconds a machine gets no work after a connection failure or two failures in a row. |
| `probe_interval` | `15` | How often static nodes are polled and unreachable agents re-checked. |
| `static_nodes` | `[]` | Model servers the coordinator calls directly, without an agent (below). |

A **static node** is any OpenAI-compatible server the coordinator can reach:

```yaml
cluster:
  static_nodes:
    - name: gpu-box
      url: http://192.168.1.50:11434    # Ollama: models are discovered via /api/tags
      kind: ollama                      # ollama | openai | agent
      max_concurrency: 4
      labels: {gpu: nvidia}
    - name: lmstudio
      url: http://192.168.1.60:1234     # plain OpenAI-compatible: models from /v1/models
      kind: openai
      api_key: ${LMSTUDIO_KEY}
      models: [qwen3.5-4b]              # optional: fixed list when discovery is not possible
```

`kind: agent` points at a Flotilla agent that does not register itself (no
`FLOTILLA_COORDINATOR_URL`); the coordinator sends it `FLOTILLA_CLUSTER_TOKEN`
unless `api_key` is set.

Prefer agents where you can: they keep the model server private, report load
and loaded models, and can download models on request.

### `defaults` and members

Members are named models with a role. Every field except `model`/`team` and
`description` can also appear under `defaults`.

```yaml
defaults:
  temperature: 0.7
  max_tokens: 1536
  reasoning_effort: none

members:
  writer:
    model: [qwen3.5:9b, gemma4:12b, qwen3.5:4b]   # first one any machine has
    system: You write clear, well-structured answers.
    description: Merges and polishes (shown to planners and routers)
    temperature: 0.4
    max_tokens: 2048
    prefer_labels: {gpu: nvidia}
  tester:
    model: granite4.2:8b
    node: gpu-box                  # always run on this machine
  specialist:
    team: council                  # a whole team acting as one member
```

| Field | Meaning |
| --- | --- |
| `model` | A model name, or a list of fallbacks in order of preference. Names match Ollama tags (`llama3.2` equals `llama3.2:latest`), vLLM `--served-model-name`, or llama.cpp `--alias`. |
| `team` | Use another team as this member (instead of `model`). |
| `system` | Persona prepended to the system prompt of every call this member makes. |
| `description` | Shown to planners (worker roster) and in the dashboard. |
| `temperature`, `top_p`, `max_tokens`, `seed`, `stop` | Passed to the model server. |
| `reasoning_effort` | Passed through. With Ollama, `none` turns thinking off on thinking models (much faster, and avoids empty answers when the token budget runs out while thinking); `low`/`medium`/`high` turn it on. If a server rejects the field, Flotilla retries without it and stops sending it to that server. |
| `timeout` | Seconds for this member's calls. |
| `node` | Only run on the machine with this name. |
| `labels` | Only run on machines whose labels match (`FLOTILLA_NODE_LABELS` on the agent). |
| `prefer_labels` | Prefer machines with these labels, use others when needed. |
| `extra` | Merged into the request body as-is, e.g. `{chat_template_kwargs: {enable_thinking: false}}` for vLLM. |

Wherever a team expects a member you can write the member's name,
`team:<name>`, or an inline definition such as `{model: qwen3.5:4b, temperature: 0}`.

### Teams

Every team has `strategy`, an optional `description`, optional `timeout`
(seconds) and optional `prompts` (template overrides, below).

#### `single`
One member, e.g. a persona with its own system prompt.
```yaml
translator: {strategy: single, member: {model: qwen3.5:4b, system: Translate into French.}}
```

#### `mixture` (mixture of agents)
```yaml
moa:
  strategy: mixture
  proposers: [qwen, granite, ministral]   # answer in parallel
  aggregator: writer                      # merges them; its answer streams to the client
  layers: 1                               # 2+: proposers see the previous layer's answers and improve on them
  min_success: 1                          # proposers that must succeed
```

#### `council`
```yaml
council:
  strategy: council
  members: [qwen, granite, ministral]     # answer in parallel
  reviewers: []                           # default: the members; each ranks all answers (anonymised)
  chairman: writer                        # sees answers, reviews and the combined (Borda) ranking
```

#### `critique`
```yaml
critique:
  strategy: critique
  writer: writer
  critics: [granite, ministral]
  max_rounds: 2                           # critique rounds; each unhappy round triggers a revision
  approve_token: APPROVED                 # critics answer exactly this to approve
```

#### `plan`
```yaml
planner:
  strategy: plan
  planner: strategist                     # returns JSON subtasks; JSON mode is requested
  workers: [qwen, granite, ministral]     # the planner assigns subtasks by name (descriptions help)
  synthesizer: writer
  max_subtasks: 5
  json_mode: true                         # set false if the model server rejects response_format
```
Subtasks can depend on earlier ones (`depends_on`); independent subtasks run in
parallel across machines. If the planner's output cannot be parsed after one
retry, the whole request runs as a single subtask.

#### `vote`
```yaml
vote:
  strategy: vote
  voters: [qwen, granite, ministral]
  samples: 1                              # answers per voter
  judge: judge                            # optional; without it the most common answer wins
```
Majority voting compares normalised text, so it suits short answers
(classifications, numbers, yes/no). Use a judge for open-ended answers.

#### `route`
```yaml
auto:
  strategy: route
  router: router                          # returns {"route": "<name>"}
  default: deep                           # used when the router's reply is unusable
  routes:
    - {name: quick, description: Short factual questions, target: qwen}
    - {name: deep,  description: Open questions needing several views, target: "team:moa"}
```

#### `pipeline`
```yaml
pipeline:
  strategy: pipeline
  stages:
    - {member: qwen, instruction: Write a first draft.}
    - {member: granite, instruction: Fix factual errors. Reply with only the corrected text.}
```
Each stage sees the original request plus the previous stage's output.

### Prompt overrides

Each strategy's prompts can be replaced per team. Placeholders use double braces.

| Key | Used by | Placeholders |
| --- | --- | --- |
| `aggregate` | mixture | `answers` |
| `refine` | mixture, layers 2+ | `answers` |
| `review` | council | `task`, `answers`, `example_ranking` |
| `chair` | council | `answers`, `reviews`, `ranking` |
| `critic` | critique | `task`, `draft`, `approve_token` |
| `revise` | critique | `feedback` |
| `plan` | plan | `max_subtasks`, `workers`, `task` |
| `work` | plan (worker system prompt) | none |
| `work_input` | plan | `task`, `title`, `instructions`, `context` |
| `synthesize` | plan | `results` |
| `judge` | vote | `task`, `answers` |
| `route` | route | `routes`, `task` |
| `stage_input` | pipeline | `previous`, `instruction` |

The defaults live in `flotilla/engine/prompts.py`. Parsers expect the same
reply formats (`FINAL RANKING: ...`, `BEST: X`, the approve token, JSON), so
keep those instructions when you rewrite a prompt.

## Environment variables

### Coordinator

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLOTILLA_CONFIG` | `config/flotilla.yaml` | Team file. |
| `FLOTILLA_CLUSTER_TOKEN` | none (required) | Shared secret agents use; also sent when calling agents. |
| `FLOTILLA_API_KEYS` | empty (no auth) | Comma-separated keys accepted from clients (`Authorization: Bearer` or `x-api-key`). |
| `FLOTILLA_HOST` / `FLOTILLA_PORT` | `0.0.0.0` / `8800` | Listen address. |
| `FLOTILLA_DATA_DIR` | none (`/data` in Docker) | Where finished task traces are appended (`tasks.jsonl`). |
| `FLOTILLA_CLUSTER_NAME` | `flotilla` | Shown in the dashboard. |
| `FLOTILLA_TLS_CERT` / `FLOTILLA_TLS_KEY` | none | Serve HTTPS. |
| `FLOTILLA_CA_FILE` | none | CA bundle for verifying agents that use TLS with a private CA. |
| `FLOTILLA_TRUST_ENV_PROXY` | `false` | Honour `HTTP(S)_PROXY` for calls to machines (off so LAN traffic never goes to a corporate proxy). |
| `FLOTILLA_LOG_LEVEL` | `info` | `debug` also logs every HTTP request. |

### Agent

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLOTILLA_COORDINATOR_URL` | none | Where to register, e.g. `http://192.168.1.10:8800`. |
| `FLOTILLA_CLUSTER_TOKEN` | none (required) | Same value as on the coordinator. |
| `FLOTILLA_NODE_NAME` | host of the advertise URL, else hostname | Unique machine name. |
| `FLOTILLA_ADVERTISE_URL` | none | URL the coordinator uses to reach this agent. When empty, the coordinator uses the heartbeat's source address and `FLOTILLA_ADVERTISE_PORT`. |
| `FLOTILLA_ADVERTISE_PORT` | `FLOTILLA_PORT` | Published host port, when it differs from the container port. |
| `FLOTILLA_PORT` | `8801` | Listen port. |
| `FLOTILLA_BACKEND` | `ollama` | `ollama`, or `openai` for vLLM, llama.cpp, LM Studio and other OpenAI-compatible servers. |
| `FLOTILLA_BACKEND_URL` | `http://127.0.0.1:11434` | The model server (without `/v1`). |
| `FLOTILLA_BACKEND_API_KEY` | none | Sent to the model server (e.g. vLLM `--api-key`). |
| `FLOTILLA_MAX_CONCURRENCY` | `2` | Parallel calls this machine accepts. Match `OLLAMA_NUM_PARALLEL`; vLLM handles more. |
| `FLOTILLA_PULL_MODELS` | none | Ollama models to download at start, comma separated. |
| `FLOTILLA_MODELS` | none | Advertise exactly these model names (for servers that cannot list models). |
| `FLOTILLA_NODE_LABELS` | none | `key=value` pairs for placement, e.g. `gpu=nvidia,vram=12`. |
| `FLOTILLA_INCLUDE_REMOTE_MODELS` | `false` | Also advertise Ollama cloud models (off: the cluster stays local). |
| `FLOTILLA_HEARTBEAT_INTERVAL` | `10` | Seconds between heartbeats. |
| `FLOTILLA_QUEUE_TIMEOUT` / `FLOTILLA_REQUEST_TIMEOUT` | `300` / `600` | Waiting for a local slot / one proxied call. |
| `FLOTILLA_TLS_CERT` / `FLOTILLA_TLS_KEY` / `FLOTILLA_CA_FILE` | none | As for the coordinator. |
