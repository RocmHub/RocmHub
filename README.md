# ROCmHub

> *"We make every open model AMD-ready automatically."*

ROCmHub is an open-source platform designed to automate the preparation, optimization, benchmarking, and publishing of open AI models on AMD GPUs (Radeon and Instinct) and the ROCm software stack.

---

## Current Status: Phase 1 (Foundation & Core Contracts)

ROCmHub is currently in active development. Phase 1 defines the core domain types, execution statuses, data contracts, and reproducible artifact manifest schemas.

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
- Linux with ROCm 6.0+ and AMD GPU (for native inference benchmarking)
- macOS / Windows / Linux (for development, schema validation, and diagnostic mode)

### Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### CLI Usage

```bash
# Display version
rocmhub --version

# Show environment (Phase 3)
rocmhub env

# Inspect model (Phase 4)
rocmhub inspect Qwen/Qwen2.5-0.5B-Instruct

# Run benchmark pipeline (Phase 5-8)
rocmhub run --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16
```

---

## License

Apache License 2.0. See LICENSE for details.
