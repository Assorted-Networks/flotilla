# Flotilla

Flotilla runs many small LLMs on several Linux machines and has them work
together on each request. Teams are defined in YAML (mixture-of-agents, peer
review, planners with parallel workers, critique loops, voting, routing and
pipelines), and each team appears to clients as one OpenAI-compatible model.
Everything is self-hosted and runs in Docker.

```
 Open WebUI / curl / OpenAI SDK / CLI / dashboard
                     |
            OpenAI-compatible API (:8800)
            +----------------------+
            |     coordinator      |  teams, scheduler, failover, traces
            +----------------------+
              |         |         |      HTTP(S) over your LAN or VPN
         +---------+ +---------+ +-----------+
         |  agent  | |  agent  | |   agent   |   one per machine (:8801)
         | Ollama  | |  vLLM   | | llama.cpp |   the model server stays private
         +---------+ +---------+ +-----------+
          machine A   machine B    machine C
```

When you ask `team/moa` a question, three small models answer in parallel on
different machines, then the largest model available anywhere in the cluster
merges their answers. Machines join by starting an agent. If a machine fails,
the calls it had not finished are retried on another machine with the same
model (the one exception is a final answer that had already started streaming).

## Why this exists

Nothing off the shelf covers all four parts (small models collaborating,
self-hosted, across machines, containerized). The existing tools each do a
piece:

| You want to... | Existing tools | What is missing |
| --- | --- | --- |
| Split **one big model** across machines | llama.cpp RPC, exo, LocalAI worker mode, GPUStack | They make one model bigger; they do not make several models collaborate. exo currently runs on CPU on Linux; llama.cpp RPC is unauthenticated and usually slower than one machine when the model already fits. |
| Spread requests over **many Ollama boxes** | Olla, LiteLLM, LocalAI federated mode, ollamaMQ | Load balancing only: each request still goes to one model. |
| Make **models collaborate** | Mixture-of-Agents ports, Karpathy's llm-council, CrewAI, LangGraph, Microsoft Agent Framework | Single-process apps or libraries; you build the multi-machine scheduling, failover and deployment yourself. llm-council targets cloud APIs through OpenRouter. |

Flotilla is the layer in between: cluster membership and load-aware
scheduling across machines, collaboration strategies on top, one API in front.
It reuses the model servers you already know (Ollama by default, or vLLM /
llama.cpp) instead of replacing them. See [Related projects](#related-projects).

## Quick start: one machine

Needs Docker with Compose v2 and about 10 GB of disk for the default models.

```bash
git clone <this repository> flotilla && cd flotilla/deploy/all-in-one
cp .env.example .env
# Random cluster token, plus one API key shared by the dashboard, Open WebUI and scripts:
KEY=$(openssl rand -hex 24); echo "API key: $KEY"
sed -i "s/^CLUSTER_TOKEN=.*/CLUSTER_TOKEN=$(openssl rand -hex 32)/; s/^API_KEYS=.*/API_KEYS=$KEY/; s/^WEBUI_API_KEY=.*/WEBUI_API_KEY=$KEY/" .env
docker compose up -d --build                  # NVIDIA GPU: docker compose -f docker-compose.yml -f ../gpu/nvidia.yml up -d --build
docker compose logs -f agent                  # first start downloads the models
```

Then open **http://localhost:8800** (the dashboard asks for the API key once),
or start Open WebUI too:

```bash
docker compose --profile ui up -d             # http://localhost:3000
```

In Open WebUI pick a model such as `team/moa`. The team's progress (which
model ran where, and for how long) shows up in the collapsible "thinking"
block above the answer.

No GPU and no downloads needed to try it: `deploy/sim` runs a simulated
three-machine cluster with fake models (see [Simulation](#simulation)).

## Adding machines

On every additional Linux machine:

```bash
git clone <this repository> flotilla && cd flotilla/deploy/node
cp .env.example .env
# Edit .env:
#   COORDINATOR_URL=http://<head machine IP>:8800
#   CLUSTER_TOKEN=<same value as on the head machine>
#   NODE_NAME=<unique name>
#   ADVERTISE_URL=http://<this machine's IP>:8801
#   PULL_MODELS=<models that fit this machine>
docker compose up -d --build                  # NVIDIA: -f docker-compose.yml -f ../gpu/nvidia.yml
                                              # AMD:    -f docker-compose.yml -f ../gpu/amd.yml
```

Within a few seconds the machine shows up in the dashboard (and in
`docker compose exec coordinator python -m flotilla nodes` on the head machine).
Open port 8800 on the head machine and 8801 on each node in your firewall.
Machines on different networks can join over Tailscale or WireGuard; use the
VPN addresses in `COORDINATOR_URL` and `ADVERTISE_URL`.

Other model servers per machine:

| Folder | Backend | Good for |
| --- | --- | --- |
| `deploy/node` | Ollama | Default. CPU, NVIDIA, AMD; many models per machine; downloads on request |
| `deploy/node-vllm` | vLLM | NVIDIA GPUs serving many parallel calls of one model |
| `deploy/node-llamacpp` | llama.cpp server | Small CPU boxes, single-board computers, Vulkan GPUs |
| `deploy/coordinator` | none | A coordinator on a machine that runs no models |

Details (GPU prerequisites, TLS, ports, sizing) are in
[docs/deployment.md](docs/deployment.md).

## Teams and strategies

The default [config/flotilla.yaml](config/flotilla.yaml) defines these teams.
Each is callable as `team/<name>`; single models are callable by their own
name and are load-balanced across machines.

| Team | Strategy | What happens | Calls |
| --- | --- | --- | --- |
| `moa` | `mixture` | 3 models answer in parallel, the strongest available model merges the best parts | 4 |
| `council` | `council` | 3 answers, anonymous peer review and ranking, a chair writes the final answer | 7 |
| `planner` | `plan` | A planner splits the task into subtasks (with dependencies); workers solve them in parallel on different machines; a writer assembles the result | 2-7 |
| `critique` | `critique` | Draft, two critics review, revise until they approve (at most two rounds) | 3-7 |
| `vote` | `vote` | 3 answers, a judge picks the best (or majority vote without a judge) | 4 |
| `pipeline` | `pipeline` | Draft, fact-check, polish; each stage on a different model | 3 |
| `auto` | `route` | A small router picks `quick` (one model), `deep` (moa), `build` (planner) or `check` (vote) | 2+ |

Teams can contain teams (`team:moa` as a member), members list fallback
models (`model: [qwen3.5:9b, qwen3.5:4b]`, first available wins), and members
can be pinned to machines or labels (`node: gpu-box`, `labels: {gpu: nvidia}`).
Edit the YAML and apply it without a restart with `flotilla reload`. The full
reference is in [docs/configuration.md](docs/configuration.md).

Teams trade speed for quality: `moa` makes four model calls and `council`
seven, so a team run takes a few seconds on GPUs and can take a minute or
more on CPU-only machines. Whether a team beats one bigger model depends on
the task; since both are exposed side by side, compare them on your own
prompts.

## Using it

**Any OpenAI client.** Base URL `http://<head>:8800/v1`, API key = one of `API_KEYS`.

```bash
curl http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "team/moa", "messages": [{"role": "user", "content": "Is RAID a backup?"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8800/v1", api_key="YOUR_KEY")
stream = client.chat.completions.create(
    model="team/planner",
    messages=[{"role": "user", "content": "Plan a home network for a 3-bedroom house."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta is None:
        continue
    progress = getattr(delta, "reasoning_content", None)   # team progress lines
    if progress:
        print(progress, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
```

Every chat response carries an `x-flotilla-task-id` header; `GET /api/tasks/<id>`
returns the full trace (each step's model, machine, time, tokens and output).

**Command line** (inside any Flotilla container, or `pip install .`):

```bash
docker compose exec coordinator python -m flotilla run planner "Plan a data backup routine" --trace
docker compose exec coordinator python -m flotilla nodes
docker compose exec coordinator python -m flotilla teams      # shows missing models per team
docker compose exec coordinator python -m flotilla pull qwen3.5:9b --node gpu-box
docker compose exec coordinator python -m flotilla reload
```

(`--url` and `--key`, or `FLOTILLA_URL` and `FLOTILLA_API_KEY`, point the CLI
at a coordinator elsewhere.)

**Dashboard** at `http://<head>:8800/`: machines with their models, load and
hardware; teams and whether their models are available; a playground that
shows each step live on a per-machine timeline; recent tasks with full traces.

The HTTP API is documented in [docs/api.md](docs/api.md).

## Choosing models

Download sizes from the Ollama library (September 2026). A machine needs
roughly the download size plus 1-3 GB for context per loaded model.

| Model | Download | Notes |
| --- | --- | --- |
| `qwen3.5:0.8b` / `2b` / `4b` / `9b` | 1.0 / 2.7 / 3.4 / 6.6 GB | Strong all-rounders; good at JSON (planner, router). Thinking model: keep `reasoning_effort: none` for speed |
| `granite4.2:3b` / `8b` | 2.2 / 5.3 GB | Careful instruction following, summaries |
| `ministral-3:3b` / `8b` / `14b` | 3.0 / 6.0 / 9.1 GB | Different training lineage: adds diversity to mixtures |
| `gemma4:12b` | 7.6 GB | Good final writer on a 12 GB+ GPU |
| `llama3.2:3b` | 2.0 GB | Older, light, fine for CPU-only boxes |
| `qwen2.5-coder:7b` | 4.7 GB | Code-focused worker (see `config/examples/coding.yaml`) |

Diversity matters more than size for mixtures and councils: three different
3-4B families catch more of each other's mistakes than three copies of one
model. Put the same small model on several machines for parallelism and
failover, and the largest model you have on the machine with the most memory:
the default `writer` member uses the first of `qwen3.5:9b`, `gemma4:12b`,
`granite4.2:8b`, `ministral-3:8b`, `qwen3.5:4b` that any machine has.

## Simulation

`deploy/sim` starts a coordinator and three machines whose "models" are a fake
server that answers after about a second with canned text (and recognizes the
planner, reviewer, critic, judge and router prompts). Use it to learn the dashboard,
wire up clients, or watch failover:

```bash
cd deploy/sim && docker compose up -d --build
# http://localhost:8800, then:
docker compose stop node-b-agent      # requests route around the missing machine
docker compose start node-b-agent     # and it rejoins
```

## Security

- Ollama, vLLM and llama.cpp are never published; only the agent can reach
  them, and every call to an agent needs the shared `CLUSTER_TOKEN`.
- Set `API_KEYS` so that clients (Open WebUI, scripts, the dashboard) must
  send a key. Without it, anyone who can reach port 8800 can use the cluster.
- Traffic is plain HTTP by default. Across untrusted networks use a VPN
  (Tailscale/WireGuard) or TLS: `FLOTILLA_TLS_CERT`/`FLOTILLA_TLS_KEY` on
  either side and `FLOTILLA_CA_FILE` for a private CA, or a reverse proxy.
- Restrict port 8801 on each machine to the coordinator's address.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Machine shows `unreachable` | The coordinator cannot open the agent's URL. Set `ADVERTISE_URL` to an address the head machine can reach and open port 8801. |
| Machine shows `offline` with "backend unavailable" | The model server is down or still loading: `docker compose logs ollama`. |
| Agent log says the token was rejected | `CLUSTER_TOKEN` differs from the coordinator's. |
| Team shows "missing models" | No online machine has any of that member's models: `flotilla pull <model>` or add it to `PULL_MODELS`. |
| Answers are cut off or ignore instructions | The prompt exceeded the context window; raise `OLLAMA_CONTEXT_LENGTH` (default 16384 here, 4096 in plain Ollama). |
| Empty answers from thinking models | Leave `reasoning_effort: none` (the default); Flotilla also retries once without thinking when an answer comes back empty. |
| Every Open WebUI chat title runs a whole team | Keep `TASK_MODEL_EXTERNAL` (set in the compose files) pointed at a single model. |

## Limitations

- One coordinator per cluster (it keeps no critical state: agents re-register
  within seconds of a restart, so run it with `restart: unless-stopped`).
- Tool/function calling is passed through for single models but not used
  inside teams; teams ignore `tools`, `n` and sampling parameters from the
  client (`max_tokens` applies to the final step).
- Agents are reached directly over HTTP, so each machine must be reachable
  from the coordinator (a VPN solves NAT).
- Model downloads through the coordinator (`flotilla pull`) work for Ollama
  nodes; vLLM and llama.cpp nodes serve the model they were started with.

## Development

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .      # 91 tests, about 20 s
```

The tests include unit tests for parsing, config, scheduling and every
strategy, plus end-to-end runs that start a coordinator, agents and fake model
servers as separate processes and kill machines mid-run.

Layout: `flotilla/coordinator.py` (API, dashboard), `flotilla/agent.py`,
`flotilla/registry.py` (membership, slots), `flotilla/dispatch.py` (failover),
`flotilla/engine/` (strategies and prompts), `flotilla/fake_backend.py`
(test double), `deploy/` (Compose files), `config/` (teams).

## Related projects

- [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) — splits one model's layers across machines.
- [exo](https://github.com/exo-explore/exo) — peer-to-peer model sharding; GPU on macOS, CPU on Linux ([summary](https://toolhalla.ai/blog/exo-framework-distributed-inference-guide-2026)).
- [LocalAI distributed inference](https://localai.io/features/distribute) — federated request routing and llama.cpp weight sharding over P2P.
- [GPUStack](https://github.com/gpustack/gpustack) — GPU cluster manager for vLLM/SGLang serving.
- [Olla](https://github.com/thushan/olla) — load balancer for multiple Ollama/OpenAI-compatible backends.
- [Mixture-of-Agents with Ollama](https://github.com/severian42/MoA-Ollama-Chat) and [llm-council](https://github.com/karpathy/llm-council) — single-host collaboration apps.
- [Microsoft Agent Framework](https://atlan.com/know/ai-agent/microsoft/agent-framework/) — the successor to AutoGen and Semantic Kernel, a library for multi-agent workflows (Ollama supported).
- [Harbor](https://github.com/av/harbor) — Docker Compose toolkit for a local LLM stack on one machine.
