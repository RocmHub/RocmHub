# Deployment guide

ROCmHub is deliberately split into a public control plane and a separately run Agent. Railway hosts the FastAPI control plane and its durable state; Cloudflare Pages hosts only the static React UI. No model weights, GPU execution, or agent credentials belong in either browser build or control-plane artifact storage.

## 1. Railway backend

This repository includes `Dockerfile` and `railway.toml`. Create a Railway service from the repository and attach a persistent Volume mounted at `/data`. Set these variables in Railway:

```text
ROCMHUB_DATA_DIR=/data
ROCMHUB_CORS_ORIGINS=https://<your-pages-project>.pages.dev
ROCMHUB_AGENT_TOKENS=<one-or-more-long-random-agent-tokens-separated-by-commas>
ROCMHUB_AGENT_HEARTBEAT_TIMEOUT_SECONDS=45
```

Railway supplies `PORT`; the image runs `rocmhub serve --host 0.0.0.0` and reads that value. The persistent directory contains `jobs.db` (SQLite WAL state) and `agent-artifacts/` (only the allowlisted, UTF-8 configuration, manifest, checksum, and launcher outputs). Do not set `ROCMHUB_DB_PATH` outside `/data` unless you provide an equivalent persistent mount.

After deployment, verify:

```bash
curl -fsS https://<railway-service-domain>/health
```

`/health` is the configured Railway health check. The backend intentionally has no wildcard CORS setting; add every production Pages origin explicitly when using previews or a custom domain.

## 2. Cloudflare Pages frontend

Create a Pages project from `frontend/` with:

```text
Build command: npm ci && npm run build
Build output directory: dist
Environment variable: VITE_API_BASE_URL=https://<railway-service-domain>
```

`frontend/public/_redirects` keeps SPA routes on `index.html`. `VITE_API_BASE_URL` is public build-time configuration; never put an Agent token in it.

## 3. Remote Agent

Run the Agent on the machine that is allowed to prepare model configurations. It needs outbound HTTPS access to Railway. The control plane never opens inbound connections to the Agent.

```bash
export ROCMHUB_AGENT_TOKEN='<same value configured in Railway>'
rocmhub agent start \
  --server https://<railway-service-domain> \
  --token "$ROCMHUB_AGENT_TOKEN" \
  --name linux-rocm-agent \
  --workspace "$PWD/rocmhub-agent-workspace" \
  --json
```

The Agent reports capability metadata, heartbeats, atomically claims only `PREPARE_MODEL_FOR_AMD`, and uploads only small allowlisted text artifacts. It does not run arbitrary command input, upload private workspace files, download model weights, or claim physical AMD execution from configuration-only work.

## 4. Production smoke test

With a valid deployment and Agent token, run this from an Agent-capable host:

```bash
python scripts/remote_agent_e2e.py --server https://<railway-service-domain>
```

The script requires `ROCMHUB_AGENT_TOKEN` (or `--token`), performs health, registration, heartbeat, queue, atomic claim, real configuration-only Forge preparation, artifact upload/download discovery, and completion checks. It creates no weights and removes its temporary local workspace unless `--keep-workspace` is supplied.

## Operations and recovery

- Back up the Railway `/data` Volume before destructive service changes; SQLite and retained small artifacts survive container restarts through that Volume.
- A restarted backend releases a stale claimed preparation job rather than claiming it completed. A reconnecting Agent re-registers with its stable server identity header and resumes heartbeats.
- Rotate Agent credentials by changing `ROCMHUB_AGENT_TOKENS`, restarting the service, then updating Agents. An Agent can also revoke its currently used token at `/api/v1/agents/revoke`.
- Keep the Railway URL and explicit CORS origin aligned with the Pages production/custom-domain URL. Browser requests use HTTPS; the Agent uses outbound HTTPS to the same API.
