# Changelog

All notable changes to the ROCmHub project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-09-23

Initial MVP release of ROCmHub — open-source tooling and platform for preparing, running, validating, benchmarking, and optimizing AI models for AMD ROCm.

### Added
- **Model Acquisition & Inspection**: Static Hugging Face repository metadata inspector, immutable 40-character Git commit SHA resolution, and architecture compatibility assessment without downloading model weights.
- **Hardware & Environment Diagnostics**: `rocmhub doctor` command and `rocmhub env` inspector checking host OS, Linux `/dev/kfd` driver, render/video user groups, `rocm-smi` presence, and PyTorch HIP runtime availability.
- **Model Forge**: Deterministic recipe planning (`ForgePlanner`), artifact materializer with disk quota checks, reproducible standalone inference launchers (`run_inference.py`), and execution harness (`rocmhub forge execute`).
- **Validation Engine**: Independent correctness evaluation gate with numeric corruption detection (`NaN`, `inf`, replacement characters), determinism verification, and exact/normalized token agreement metrics.
- **Benchmark Harness & Benchmark Guard**: Isolated metrics calculation (TTFT, ITL percentiles, throughput, peak memory) coupled with statistical reproducibility gating via Median Absolute Deviation (MAD) and environment fingerprinting. Diagnostic runs cleanly output `NOT_MEASURED`.
- **Reproducible Artifact Builder**: Atomic directory commits, canonical JSON serialization, fail-closed secret scanner whitelisting token metrics, and cryptographic SHA256 integrity verification (`manifest.json`, `checksums.json`).
- **Autonomous AI Engineer**: Bounded autonomous model preparation agent with budget constraints (attempts, execution time, disk allowance), tool-based trajectory logging, and executive summary reporting.
- **Optimization Lab**: Multi-candidate recipe exploration (BF16, FP16, FP32, TorchCompile) evaluated against immutable baselines with truthful non-AMD fallback semantics.
- **Backend API & Job Orchestrator**: FastAPI application (`rocmhub serve`) with FIFO SQLite job persistence (WAL mode), real-time Server-Sent Events (SSE) streaming, directory locking, cooperative job cancellation, and startup crash recovery.
- **Frontend Web Application**: Responsive React 18 / TypeScript / Vite / Tailwind CSS workspace featuring Dashboard telemetry, Model Explorer, Forge Studio, AI Engineer workspace, Optimization Lab, mobile navigation drawer, and URL hash routing.
- **CLI Suite**: Unified `rocmhub` command line interface (`--version`, `doctor`, `env`, `inspect`, `run`, `validate`, `benchmark`, `artifact`, `guard`, `forge`, `engineer`, `optimize`, `serve`).
- **Automated Verification**: Multi-version CI workflow (Python 3.9–3.12, Node 20), automated fresh environment wheel installation smoke test (`scripts/fresh_install_smoke.py`), and Google Chrome real browser E2E test (`scripts/browser_e2e.js`).

### Safety & Integrity
- **Strict 5-Fact Separation for Execution**: Status `EXECUTED` and property `amd_validated=True` strictly require five independent facts: process success, inference execution confirmed via JSON payload, PyTorch HIP runtime used, physical AMD GPU hardware confirmed via HIP device query, and output validation passed.
- **Zero Synthetic Metrics**: Diagnostic and non-ROCm runs evaluate strictly to `CONFIG_ONLY`, `PREPARED`, `NO_ACCELERATOR`, or `NOT_MEASURED`. Zero fake latency, throughput, or VRAM numbers are ever generated.
- **Security Boundaries**: Path traversal prevention on job output directories (workspaces roots cannot be used directly as output dirs), strict model identifier validation regex, and sanitized HTTP 500 error responses preventing internal stack trace or secret leakage.

### Developer Experience
- **Preflight Fail-Closed Behavior**: Commands exit with semantic codes (`0=PASS`, `2=NO_ACCELERATOR / NOT_MEASURED`, `3=BLOCKED / FAIL`, `4=UNKNOWN / INCONCLUSIVE`, `1=ERROR`).
- **Safe Dry-Run Modes**: Added `--dry-run` flag to `rocmhub run` and `rocmhub forge execute`, permitting risk-free pipeline verification on developer laptops without downloading multi-gigabyte model weights.
- **Packaging**: Standard `pyproject.toml` with PEP 561 `py.typed` typing marker and Apache-2.0 license.

### Known Limitations
- **No Physical AMD Hardware Validation Yet**: All development, unit testing, and E2E browser validation have been conducted on non-AMD environments (Apple Silicon Mac and Linux CPU CI). Physical GPU execution, `/dev/kfd` kernel driver calls, and HIP tensor allocations have **not yet been physically validated on real AMD hardware**.
- **Target First Hardware Milestone**: The first physical hardware validation run is targeted for an AMD Radeon RX 7900 XTX (`gfx1100`) running ROCm on Linux with `Qwen/Qwen2.5-0.5B-Instruct` (see `FIRST_AMD_RUN.md`).
- **Single Host / Local Scope**: Multi-node cluster orchestration, distributed multi-GPU training, remote model registries, and authentication are out of scope for the 0.1.0 MVP.
