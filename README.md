# ROCmHub

## External Agent quickstart

The control plane keeps lightweight job state; model preparation runs on a separately started ROCmHub Agent. The same protocol works on a macOS preparation host and a future Linux ROCm host.

```bash
# Terminal A — control plane (set a non-default value outside local development)
export ROCMHUB_AGENT_TOKENS='replace-with-a-long-random-token'
export ROCMHUB_DB_PATH="$PWD/.rocmhub/jobs.db"
rocmhub serve --host 127.0.0.1 --port 8000

# Terminal B — frontend
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev

# Terminal C — agent. The token is never printed by the Agent.
rocmhub agent start --server http://127.0.0.1:8000 --token "$ROCMHUB_AGENT_TOKENS" --name mac-dev-agent
```

On macOS, `PREPARE_MODEL_FOR_AMD` is intentionally `CONFIG_ONLY` or `PREPARED`: it writes reproducible configuration, launcher, manifest, and hashes but does not claim AMD execution, validation, benchmarks, or download full weights. The Agent accepts only predefined ROCmHub jobs and never runs arbitrary shell commands from job input.

For deployment, configure `PORT`, `ROCMHUB_DB_PATH`, `ROCMHUB_CORS_ORIGINS`, and `ROCMHUB_AGENT_TOKENS` through the environment. The frontend uses `VITE_API_BASE_URL` when it is set, and retains same-origin local development by default.

> Open-source tooling and engineering platform for preparing, benchmarking, validating, and optimizing AI models for AMD ROCm.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](pyproject.toml)
[![Release](https://img.shields.io/badge/Release-0.1.0_MVP-green)](RELEASE_NOTES_0.1.0.md)

---

## What is ROCmHub?

ROCmHub is an open-source engineering platform designed to automate and standardize the process of preparing, running, verifying, benchmarking, and optimizing open AI models (such as Hugging Face CausalLM architectures) for AMD GPUs (Radeon and Instinct) and the AMD ROCm software stack.

ROCmHub provides a disciplined, reproducible engineering harness with:
- **Deterministic Build Planning**: Pre-flight validation, disk quotas, and standalone executable launch scripts;
- **Strict Execution Provenance**: Immutable 40-character Git commit SHAs, exact GPU device telemetry, and `gfx` targets;
- **Truthful Status Reporting**: Zero synthetic metrics. When a GPU measurement cannot be taken, the status evaluates cleanly to `NOT_MEASURED` or `CONFIG_ONLY`;
- **Output Quality & Correctness Gates**: Numeric corruption detection (`NaN`, `inf`, replacement characters) and token agreement evaluation;
- **Statistical Reproducibility**: Benchmark Guard with Median Absolute Deviation (MAD) dispersion checks;
- **Autonomous AI Engineer**: Bounded optimization agent for automated iteration and failure recovery;
- **Interactive Workspaces**: Unified CLI, local REST API, and modern reactive web application.

> [!NOTE]
> ROCmHub is an independent open-source developer tool. It is **not** an official AMD product and does not claim official hardware certification.

---

## Current MVP Capabilities: Implemented vs. Physically Validated

To maintain strict scientific and technical transparency, we clearly delineate between what is **implemented in code / tested in CI** and what is **physically validated on AMD hardware**:

| Capability | Status | Implementation Details |
|---|---|---|
| **Model Metadata Inspection** | **IMPLEMENTED** | Hugging Face Hub metadata, config params, architecture check, and immutable 40-char commit SHA resolution without downloading weights. |
| **Hardware & Environment Diagnostics** | **IMPLEMENTED** | `rocmhub doctor` checks OS, `/dev/kfd` access, user groups (`render`, `video`), `rocm-smi`, GPU detection, and PyTorch HIP runtime. |
| **Model Forge (Planning & Materialization)** | **IMPLEMENTED** | `ForgePlanner` generates deterministic recipes (`pytorch_transformers_hip`), verified build manifests, and standalone `run_inference.py` launchers. |
| **Safe CONFIG_ONLY Mode** | **IMPLEMENTED** | Generates all runtime scripts and configs without downloading weight tensors or executing inference. |
| **AMD Execution Harness** | **IMPLEMENTED** | Subprocess execution harness with stdout/stderr capture, timeout enforcement, and structured JSON metrics reporting. |
| **Validation Engine** | **IMPLEMENTED** | Output corruption gate (rejects `NaN`, `inf`, `\ufffd`), determinism test, and exact/normalized token agreement scoring. |
| **Benchmark Harness & Guard** | **IMPLEMENTED** | Latency, throughput, TTFT, and ITL percentiles calculation. `BenchmarkGuard` validates MAD variability and environment drift. |
| **Reproducible Artifact Builder** | **IMPLEMENTED** | Atomic POSIX directory commits, canonical JSON serialization, secret scanner, and SHA256 detached checksums. |
| **Autonomous AI Engineer** | **IMPLEMENTED** | Bounded autonomous agent (`BASE_PREPARATION`) operating under strict budget bounds (attempts, timeout, disk usage). |
| **Optimization Lab** | **IMPLEMENTED** | Multi-candidate recipe exploration (BF16, FP16, FP32, TorchCompile) evaluated against immutable baselines. |
| **FastAPI Backend & Orchestrator** | **IMPLEMENTED** | Async job queue (SQLite WAL mode), Server-Sent Events (SSE) live streaming, directory locking, and crash recovery. |
| **Modern Web Application** | **IMPLEMENTED** | Single-page UI (React 18, TypeScript, Tailwind CSS) with real-time SSE event viewer, hardware telemetry, and URL hash routing. |
| **Unified CLI** | **IMPLEMENTED** | `rocmhub` command line interface covering all core domains with semantic exit codes. |
| **Automated Verification Pipeline** | **IMPLEMENTED** | 368 Pytest unit tests, Ruff, Mypy, fresh environment wheel smoke test, and Google Chrome E2E browser automation. |
| **Physical AMD Hardware Execution** | **TARGET / PENDING** | Physical `/dev/kfd` kernel ioctl calls, HIP memory allocation, and kernel execution on physical AMD silicon **have not yet been executed**. |

---

## Current Hardware Status & Target Milestone

### Current Host Testing
- **Development & Local Verification**: Verified on Apple Silicon Mac (macOS Darwin arm64) in diagnostic and dry-run modes (`NO_ACCELERATOR`, `CONFIG_ONLY`, `NOT_MEASURED`).
- **Continuous Integration**: Verified on Ubuntu Linux runners (Python 3.9–3.12, Node 20) without GPU accelerators.
- **Safety Invariant**: Under non-AMD hosts, the tool gracefully degrades. It **never** downloads weights unexpectedly, **never** launches unverified inference, and **never** generates fake performance numbers.

### Target First AMD Run
The primary target configuration prepared for the initial physical hardware validation milestone:
- **Target GPU**: AMD Radeon RX 7900 XTX (24 GB VRAM)
- **Target Architecture**: `gfx1100` (RDNA 3)
- **Host OS**: Ubuntu 22.04 LTS (recommended) / Ubuntu 24.04 LTS (target configuration to verify before run)
- **ROCm Stack**: ROCm 6.2+
- **PyTorch**: Official PyTorch ROCm build (`torch.version.hip` active)
- **Target Model**: `Qwen/Qwen2.5-0.5B-Instruct` (FP16)

Detailed preflight verification and step-by-step instructions for physical execution are documented in [FIRST_AMD_RUN.md](FIRST_AMD_RUN.md).

---

## Quick Start

### 1. Clone & Setup Python Environment

```bash
git clone <repository-url>
cd ROCmHub

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
```

### 2. Verify Installation

```bash
# Check version (0.1.0)
rocmhub --version

# Run hardware and environment doctor
rocmhub doctor
```

### 3. Safe No-GPU Workflow (Mac / CPU / Laptop)

All of the following commands run safely without requiring an AMD GPU or downloading model weight tensors:

```bash
# 1. Inspect model metadata and resolve immutable commit SHA
rocmhub inspect Qwen/Qwen2.5-0.5B-Instruct

# 2. Forge a deterministic build in CONFIG_ONLY mode (no weights downloaded)
rocmhub forge build Qwen/Qwen2.5-0.5B-Instruct --output-dir builds/qwen2.5 --no-weights

# 3. Dry-run verify the generated launcher without executing subprocess
rocmhub forge execute builds/qwen2.5 --dry-run

# 4. Dry-run verify benchmark preflight (returns exit code 2: NO_ACCELERATOR)
rocmhub run --model Qwen/Qwen2.5-0.5B-Instruct --dry-run

# 5. Run the automated interactive demo tour
python3 scripts/mvp_demo_tour.py
```

### 4. Launch Local Web Platform

Start the local FastAPI backend API server:
```bash
rocmhub serve --host 127.0.0.1 --port 8000
```

In a separate terminal, start the frontend web development server:
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser to access the ROCmHub workspace.

---

## Repository Structure

```text
ROCmHub/
├── src/rocmhub/
│   ├── models/        # Model source adapters, Hugging Face metadata inspection, immutable commit SHA resolution.
│   ├── hardware/      # Environment observation, /dev/kfd driver check, ROCm stack and GPU detection.
│   ├── runners/       # Process execution runners and isolated Hugging Face inference runtime adapters.
│   ├── forge/         # Deterministic build planning, materialization, recipe templates, and standalone launcher.
│   ├── validation/    # Output correctness gating, determinism, token agreement, and quality retention evaluation.
│   ├── benchmarks/    # Latency, throughput, TTFT, and ITL metrics calculation without synthetic numbers.
│   ├── guard/         # Statistical reproducibility gate, Median Absolute Deviation (MAD), and health checks.
│   ├── artifacts/     # Cryptographic hashing, secret scanning, tamper verification, and atomic artifact store.
│   ├── engineer/      # Autonomous AI Engineer agent with budget bounds, tool execution, and trajectory reporting.
│   ├── optimization/  # Multi-candidate exploration, compilation recipes, and baseline comparison engine.
│   ├── server/        # FastAPI REST backend, SQLite job queue, SSE live streaming, and cooperative cancellation.
│   ├── doctor.py      # System diagnostics checker (kernel, permissions, rocm-smi, torch.version.hip).
│   └── cli/           # Unified command line interface (rocmhub ...).
├── frontend/          # Modern responsive web application (React 18, TypeScript, Tailwind CSS, TanStack Query).
├── tests/             # Comprehensive automated test suites (368+ unit, safety invariant, and API tests).
├── scripts/           # Reproducible scenarios, fresh environment smoke tests, and Chrome E2E browser automation.
├── examples/          # Declarative configuration examples (qwen2.5-0.5b-fp16.json).
└── screenshots/       # Verified Google Chrome E2E browser acceptance screenshots.
```

---

## Documentation Index

- [ARCHITECTURE.md](ARCHITECTURE.md) — Comprehensive technical architecture, design invariants, and module contracts.
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) — Complete roadmap and implementation log across Phases 1–19.
- [FIRST_AMD_RUN.md](FIRST_AMD_RUN.md) — Step-by-step guide for performing the first physical AMD GPU validation run.
- [RELEASE_NOTES_0.1.0.md](RELEASE_NOTES_0.1.0.md) — Detailed overview of capabilities, testing boundaries, and limits for the 0.1.0 MVP.
- [CHANGELOG.md](CHANGELOG.md) — Version history and changes following Keep a Changelog.
- [CONTRIBUTING.md](CONTRIBUTING.md) — Developer setup, quality gates, and the Hardware Truthfulness rule.
- [SECURITY.md](SECURITY.md) — Security policy and vulnerability disclosure procedures.
- [LICENSE](LICENSE) — Apache License 2.0.

---

## Running Verification & Quality Gates

```bash
# Run backend test suite (368+ tests)
pytest

# Python linting and type checking
ruff check src tests scripts
mypy src tests/test_api.py

# Package build & fresh virtual environment installation smoke test
python -m pip install build && python -m build
python3 scripts/fresh_install_smoke.py

# Frontend tests and production build
cd frontend && npm test -- --run && npm run build && cd ..

# Google Chrome real browser E2E acceptance test
node scripts/browser_e2e.js
```

---

## License

ROCmHub is licensed under the [Apache License 2.0](LICENSE).
