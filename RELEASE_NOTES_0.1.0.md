# ROCmHub 0.1.0 Release Notes

Welcome to the initial Minimum Viable Product (MVP) release of **ROCmHub** (`v0.1.0`).

ROCmHub is an open-source engineering platform built to streamline and automate the process of preparing, benchmarking, validating, and optimizing open AI models for AMD GPUs (Radeon and Instinct) using the AMD ROCm software stack.

---

## 1. What is ROCmHub 0.1.0

ROCmHub bridges the gap between open-weight machine learning models (such as Hugging Face CausalLM architectures) and AMD GPU hardware. Rather than treating ROCm deployment as an ad-hoc trial-and-error process, ROCmHub introduces a disciplined, reproducible engineering harness with:

- Deterministic artifact generation;
- Statistical reproducibility gates;
- Output correctness validation;
- Truthful reporting (zero synthetic metrics);
- An autonomous AI Engineer agent for iterative preparation loops;
- A local REST API and reactive web dashboard.

---

## 2. What Works

The following components are fully implemented, integrated, and functioning in ROCmHub 0.1.0:

- **CLI (`rocmhub`)**: Full command suite including `doctor`, `env`, `inspect`, `run`, `validate`, `benchmark`, `artifact`, `guard`, `forge`, `engineer`, `optimize`, and `serve`.
- **System Doctor (`rocmhub doctor`)**: Diagnostics checking OS, kernel `/dev/kfd` device nodes, user group memberships (`render`, `video`), `rocm-smi`, GPU detection, and PyTorch HIP runtime availability.
- **Model Inspector**: Remote metadata inspection for Hugging Face models, resolving branch/tag names to immutable 40-character Git commit SHAs, inspecting config parameters, and checking causal LM architecture support without downloading multi-gigabyte weights.
- **Forge Studio & Build Engine**: Deterministic build plan generation (`ForgePlanner`), artifact materializer with disk space checks, generation of standalone executable runners (`run_inference.py`), and execution harness (`rocmhub forge execute`).
- **Validation Engine**: Independent quality and correctness evaluation verifying non-corrupted text (rejecting `NaN`, `inf`, Unicode replacement character `\ufffd`), checking determinism, and calculating exact and normalized token agreement.
- **Benchmark Harness & Guard**: Calculation of latency, TTFT, ITL percentiles, throughput, and memory without synthetic extrapolation. `BenchmarkGuard` statistically evaluates runs via Median Absolute Deviation (MAD) and environment fingerprinting.
- **Artifact Store**: Cryptographically verified, self-contained experiment bundles with canonical JSON serialization, secret scrubbing, per-file SHA256 digests, and tamper verification (`manifest.json`, `checksums.json`).
- **Autonomous AI Engineer**: Goal-driven loop (`BASE_PREPARATION`) operating under strict budget bounds (max attempts, timeout, disk limit) to diagnose and prepare models with structured action logging.
- **Optimization Lab**: Multi-candidate exploration (FP16, BF16, FP32, TorchCompile) evaluated against immutable baselines.
- **Local API Server (`rocmhub serve`)**: FastAPI backend with SQLite job queue (WAL mode), SSE live streaming, cooperative cancellation, directory locking, and crash recovery.
- **Modern Web Application**: Single-page application built with React 18, TypeScript, Vite, and Tailwind CSS providing unified access to all platform studios.

---

## 3. What Has Been Tested

The ROCmHub 0.1.0 codebase has undergone rigorous multi-layer testing:

- **Automated Unit & Integration Test Suite**: 368 passing Pytest tests covering data structures, command-line parsers, benchmark math, validation metrics, artifact hashing, security boundaries, and API routes.
- **Frontend Quality**: 12 passing Vitest component tests and clean TypeScript production bundle compilation (`tsc && vite build`).
- **Multi-Version CI Matrix**: GitHub Actions workflow verifying Python 3.9, 3.10, 3.11, and 3.12 under Linux runners with Ruff linting, Mypy type-checking, and full test runs.
- **Real Browser E2E Acceptance**: Automated Google Chrome test (`scripts/browser_e2e.js`) driving the live FastAPI backend and Vite frontend across desktop (1280x860) and mobile (375x812) viewports.
- **Fresh Wheel Installation Smoke Test**: Automated script (`scripts/fresh_install_smoke.py`) testing package wheel build, clean venv installation, entrypoints, and non-AMD dry-run behavior.
- **Local Developer Acceptance**: Verified on macOS (Apple Silicon Darwin arm64) in diagnostic and dry-run modes.

---

## 4. What Has NOT Been Validated Yet

To maintain scientific integrity and transparency, we explicitly declare what has **not** been done:

> [!WARNING]
> **No Physical AMD Hardware Execution Has Taken Place Yet.**
> 
> All development, unit tests, E2E browser tests, and CI pipelines have run either on macOS (Darwin arm64) or on Linux CPU runners without an AMD GPU accelerator.
> 
> The hardware execution contracts, `/dev/kfd` calls, HIP runtime initializations, and GPU kernel executions are fully implemented in software, but **have not yet been physically validated on an actual physical AMD GPU machine**.

---

## 5. First AMD Target Milestone

The first target configuration planned for physical validation is:

- **GPU**: AMD Radeon RX 7900 XTX (24 GB VRAM)
- **Architecture**: `gfx1100` (RDNA 3)
- **Operating System**: Ubuntu 22.04 LTS (recommended) or Ubuntu 24.04 LTS (target configuration to verify before run)
- **ROCm Stack**: ROCm 6.2+
- **PyTorch**: Official PyTorch ROCm/HIP build (`torch.version.hip` active)
- **Target Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Target Precision**: `fp16`

Step-by-step instructions for performing this run are documented in [FIRST_AMD_RUN.md](FIRST_AMD_RUN.md).

---

## 6. Known Limitations

- **Single Host Only**: Designed as a local developer and workstation platform; no multi-node clustering or distributed orchestration.
- **PyTorch Transformers Runtime**: Current recipe focuses on standard PyTorch + Hugging Face Transformers. Dedicated vLLM, SGLang, and specialized quantization runtimes are planned for future phases.
- **Local Storage**: Job states and artifacts are persisted in local SQLite and filesystem directories; no remote cloud registry or S3 integration is included in 0.1.0.
- **Diagnostic Fallback**: When run without an AMD GPU, the tool operates strictly in diagnostic mode (`CONFIG_ONLY`, `NO_ACCELERATOR`, `NOT_MEASURED`).

---

## 7. Next Milestone

Following the release of 0.1.0, the immediate next milestone is:
- **Physical Hardware Execution Validation**: Running `scripts/first_amd_run.py` on a physical AMD Radeon RX 7900 XTX / ROCm host, capturing real hardware logs, and recording the first authentic `EXECUTED` manifest.
