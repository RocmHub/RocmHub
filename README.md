# ROCmHub

> *"We make every open model AMD-ready automatically."*

ROCmHub is an open-source platform designed to automate the preparation, optimization, benchmarking, and publishing of open AI models on AMD GPUs (Radeon and Instinct) and the ROCm software stack.

---

## Current Status: Phase 15 (Product Acceptance & Polish Complete)

ROCmHub features a complete platform stack validated by real Google Chrome E2E browser acceptance testing:
- **Core Domain Engine**: Model acquisition, static inspection, hardware detection, Benchmark Guard, and reproducible artifact builder.
- **Model Forge**: Automated preparation and reproducible recipe generation for open causal language models (`pytorch_transformers_hip`).
- **Autonomous AI Engineer**: Bounded agentic loop for AMD GPU model optimization and failure recovery with structured trajectory and executive reports.
- **Optimization Engine**: Multi-candidate generation, compilation, execution, and objective comparison against immutable baselines (truthful `NOT_MEASURED` semantics on non-ROCm hosts).
- **Backend API & Job Orchestrator**: FastAPI server (`rocmhub serve`) with FIFO SQLite job queue, SSE event streaming, directory locking, cooperative cancellation, and crash recovery.
- **Frontend Workspace**: Responsive React 18 / TypeScript / Vite / Tailwind CSS web application featuring mobile navigation drawer, real-time SSE log streaming, live hardware telemetry, deep linking, model explorer, forge studio, AI engineer workspace, and optimization lab.

For architectural decisions, principles, and roadmap, see:
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

---

## Key Principles

1. **Deterministic Reproducibility**: Full provenance for every run (immutable commit SHA, exact GPU, open `gfx` target, ROCm/HIP versions, driver, and environment variables).
2. **Benchmark Guard Integrity**: Benchmarking is isolated from optimization. Diagnostic mode on non-ROCm machines never generates synthetic or fake performance metrics.
3. **Runtime-Agnostic Core**: Inference engines are decoupled behind adapters (PyTorch baseline, with future vLLM / SGLang integration).
4. **Hardware Heterogeneity**: First-class support for both AMD Instinct (CDNA) and AMD Radeon (RDNA) without hardcoded vendor assumptions.

---

## Installation & Development

### Prerequisites
- Python 3.9+
- Node.js 18+ and npm
- Linux with ROCm 6.0+ and AMD GPU (for native inference benchmarking)
- macOS / Windows / Linux (for development, schema validation, and diagnostic mode)

### Backend Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Running the Web Platform

Start the backend API server:
```bash
rocmhub serve --host 127.0.0.1 --port 8000
```

In a separate terminal, launch the frontend development server:
```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser to access the ROCmHub workspace.

---

## Running Tests

### Backend Tests & Verification
```bash
# Run all unit tests
pytest

# Linting and Type Checking
ruff check src tests
mypy src tests/test_api.py

# End-to-end integration smoke test
python3 scripts/live_demo_smoke.py
```

### Frontend Tests & Build
```bash
cd frontend
npm test
npm run build
```

### Real Browser E2E Acceptance Test
```bash
# Runs full end-to-end user scenario across desktop and mobile in Google Chrome
node scripts/browser_e2e.js
```

---

## CLI Usage

```bash
# Display version
rocmhub --version

# Show environment
rocmhub env

# Inspect model metadata
rocmhub inspect Qwen/Qwen2.5-0.5B-Instruct

# Run benchmark pipeline
rocmhub run --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16

# Start local API server
rocmhub serve --host 127.0.0.1 --port 8000
```

---

## License

Apache License 2.0. See LICENSE for details.
