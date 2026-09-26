# Deployment

## Topology

- **One coordinator** (port 8800): serves the API and dashboard, keeps the
  list of machines, runs the teams. It needs little CPU and no GPU; put it on
  any always-on machine (it can share a machine with models: `deploy/all-in-one`).
- **One agent per machine** (port 8801) next to that machine's model server.
  Agents register themselves, so adding a machine never requires editing the
  coordinator.

Traffic: agents call the coordinator (heartbeats every 10 s), and the
coordinator calls agents (model requests). Both directions must work:

| From | To | Port | Purpose |
| --- | --- | --- | --- |
| clients | coordinator | 8800 | API, dashboard |
| agents | coordinator | 8800 | registration, heartbeats |
| coordinator | agents | 8801 | model calls, model downloads |
| agent | its model server | internal | never published |

Example firewall rule on a node (ufw): `ufw allow from 192.168.1.10 to any port 8801 proto tcp`.

### Machines on different networks

The coordinator must be able to open a connection to every agent. Across the
internet or behind NAT, use a VPN such as Tailscale or WireGuard on each
host and use the VPN addresses:

```
COORDINATOR_URL=http://100.101.102.103:8800      # coordinator's tailnet address
ADVERTISE_URL=http://100.104.105.106:8801        # this node's tailnet address
```

The dashboard marks a machine `unreachable` when its heartbeats arrive but
the coordinator cannot reach the advertised URL, and the agent logs the same.

### TLS

Traffic is plain HTTP by default, which is fine on a trusted LAN or inside a
VPN. Otherwise either:

- terminate TLS in a reverse proxy (Caddy, nginx, Traefik) in front of the
  coordinator and each agent, or
- use the built-in TLS: mount a certificate and set `TLS_CERT`/`TLS_KEY` in the
  `.env` files (paths inside the container), use `https://` in
  `COORDINATOR_URL`/`ADVERTISE_URL`, and set `CA_FILE` on the other side when
  the certificates come from a private CA.

A private CA for a home lab:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.crt -days 3650 -subj "/CN=flotilla-ca"
# per machine (replace the IP):
openssl req -newkey rsa:2048 -nodes -keyout node.key -out node.csr -subj "/CN=192.168.1.21"
printf "subjectAltName=IP:192.168.1.21\n" > ext.cnf
openssl x509 -req -in node.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out node.crt -days 825 -extfile ext.cnf
```

## GPUs

### NVIDIA

Install the NVIDIA driver and the NVIDIA Container Toolkit on the host, check
with `docker run --rm --gpus all ubuntu nvidia-smi`, then start the node with
the override file:

```bash
docker compose -f docker-compose.yml -f ../gpu/nvidia.yml up -d --build
```

### AMD

Install ROCm-capable drivers on the host, then:

```bash
docker compose -f docker-compose.yml -f ../gpu/amd.yml up -d --build
```

This switches to the `ollama/ollama:rocm` image and passes `/dev/kfd` and
`/dev/dri` through. If Ollama cannot open the GPU, add the host's `video` and
`render` group ids as `group_add` (see the comment in `deploy/gpu/amd.yml`).

### Apple Silicon and Windows

Macs cannot pass the GPU into Docker. Run Ollama natively on the Mac and
point either an agent (`FLOTILLA_BACKEND_URL=http://host.docker.internal:11434`)
or a static node at it (`OLLAMA_HOST=0.0.0.0` on the Mac, then list it under
`cluster.static_nodes`). The same works for Windows machines.

## Sizing Ollama

These settings live in each node's `.env`:

| Setting | Default here | Notes |
| --- | --- | --- |
| `OLLAMA_CONTEXT_LENGTH` | 16384 | Aggregators and chairs see several answers at once; Ollama's own default (4096) silently truncates such prompts. |
| `OLLAMA_NUM_PARALLEL` | 2 | Parallel requests per loaded model. Memory for context grows with `NUM_PARALLEL x CONTEXT_LENGTH`. |
| `OLLAMA_MAX_LOADED_MODELS` | 3 | Models kept in memory at once. On a GPU, all loaded models must fit in VRAM together. |
| `OLLAMA_KEEP_ALIVE` | 30m | How long an idle model stays loaded. Loading a model takes seconds to tens of seconds. |
| `MAX_CONCURRENCY` (agent) | 2 | Parallel calls the coordinator sends this machine. |

Rules of thumb:

- A model needs roughly its download size plus 1-3 GB for context.
- Give each machine a few models and keep them loaded; the scheduler prefers
  machines where a model is already in memory, which avoids load delays.
- For mixtures, spread *different* model families across machines; for
  throughput, put the same small model on several machines.
- Put the member that writes final answers on your largest machine
  (`prefer_labels: {gpu: nvidia}` or `node: <name>`).
- CPU-only machines work; expect a team run to take a minute or more. See
  `config/examples/cpu-small.yaml`.

## vLLM and llama.cpp nodes

`deploy/node-vllm` serves one Hugging Face model with vLLM on NVIDIA GPUs and
handles many parallel calls (set `MAX_CONCURRENCY` accordingly, 8 by default).
`deploy/node-llamacpp` serves one GGUF model with llama.cpp's server, which
suits small CPU machines (`GPU_LAYERS` and a `-cuda`/`-rocm`/`-vulkan` image
use a GPU).

These servers expose one model under the name you give it (`SERVED_NAME`).
Reference that name in the team file, usually as a fallback next to the
Ollama name of a similar model:

```yaml
members:
  qwen:
    model: [qwen3.5:4b, qwen3-4b-vllm]
```

## Images

Each machine builds the Flotilla image from the repository (`--build`). To
avoid cloning the repository everywhere, build once and copy the image:

```bash
docker build -t flotilla:local .
docker save flotilla:local | ssh node-1 docker load
```

or push it to a registry you run, and replace `flotilla:local` in the compose
files. Pin `OLLAMA_IMAGE`, `VLLM_IMAGE`, `LLAMACPP_IMAGE` and `OPEN_WEBUI_IMAGE`
to specific tags once you have a working setup.

## Operating

| Task | Command |
| --- | --- |
| Machines and their state | `flotilla nodes` or the dashboard |
| Teams and missing models | `flotilla teams` |
| Download a model on some or all machines | `flotilla pull qwen3.5:9b [--node name]` |
| Apply team changes | edit `config/flotilla.yaml`, then `flotilla reload` |
| Follow a run | `flotilla run <team> "<prompt>" --trace` |
| Metrics | `GET /metrics` (Prometheus format: node up/in-flight/capacity, calls, seconds, tokens) |
| Logs | `docker compose logs -f coordinator` / `agent` |

Run the CLI inside a container (`docker compose exec coordinator python -m flotilla ...`)
or install it anywhere with `pip install .` and set `FLOTILLA_URL` and `FLOTILLA_API_KEY`.

Upgrading: pull the repository, `docker compose up -d --build` on each
machine. Agents re-register within seconds after the coordinator restarts.
Requests running on the coordinator during its restart are lost (clients see
a dropped connection). Calls running on a restarting node are retried on
another node, unless that call was already streaming the final answer.

Backups: the coordinator keeps nothing essential. `/data/tasks.jsonl` holds
past traces, and the Ollama volume holds downloaded models.
