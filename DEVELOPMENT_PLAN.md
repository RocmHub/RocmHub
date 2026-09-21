# ROCmHub Development Plan: Vertical Slice 1

> **Objective**: Implement and verify the minimal end-to-end vertical slice of ROCmHub with zero overengineering, proving the pipeline from model resolution to reproducible AMD-ready artifact generation.

---

## 1. Vertical Slice 1 Scope & Target

### Selected Model
- **Model ID**: `Qwen/Qwen2.5-0.5B-Instruct` (or fallback `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- **Rationale**:
  - Small footprint (~500M parameters, ~1GB FP16 weights).
  - Modern architecture (Qwen2/Llama-like causal LM with GQA, RoPE, RMSNorm).
  - Native `safetensors` format.
  - Downloads and runs quickly on modest VRAM (consumer Radeon, Instinct, or CPU/mock dev environments).

### Vertical Slice Pipeline
```
rocmhub run --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16
   │
   ▼
1. Model Resolution & Download (HF Hub cached)
   │
   ▼
2. Model Inspection (config.json, arch, parameter count, context length)
   │
   ▼
3. Hardware & ROCm Environment Detection (GPU, gfx target, VRAM, ROCm version, or Diagnostic mode)
   │
   ▼
4. Baseline Runner Execution (PyTorch HIP / HF Transformers)
   │
   ▼
5. Benchmark Execution (Warmup, TTFT, ITL percentiles, throughput, peak VRAM)
   │
   ▼
6. Artifact Packaging (manifest.json, metrics.json, reproduce.sh)
   │
   ▼
7. Terminal Summary Output
```

---

## 2. Implementation Stages (Step-by-Step)

### Phase 1: Project Foundation & Tooling
- [ ] Create repository base: `.gitignore`, `pyproject.toml`, `README.md`.
- [ ] Define minimal dependencies:
  - `torch` (with ROCm support on AMD systems, standard on dev/CI).
  - `transformers`, `huggingface_hub`, `safetensors`.
  - `pydantic` (for robust schemas, serialization, and validation).
  - `rich` / `typer` or standard `argparse` for clean CLI output.
  - `pytest` for automated testing.
- **Verification**: `pytest` runs and passes with basic smoke test.

### Phase 2: Core Domain Types & Schemas
- [ ] Implement `rocmhub/core/types.py`:
  - `ExecutionStatus`: Enum (`SUCCESS`, `FAILED`, `SKIPPED`, `NOT_MEASURED`).
  - `ModelSpec`: `schema_version`, model identifier, requested revision, immutable resolved `commit_sha`, architecture, parameter count.
  - `HardwareSpec`: `schema_version`, `gpu_present`, `gpu_vendor`, `device_id`, `device_name`, `family`, open `gfx_target: str | None`, `vram_total_mb`.
  - `EnvironmentSpec`: `schema_version`, OS, kernel, `rocm_version` (nullable), `hip_version` (nullable), PyTorch version, active flags.
  - `ExperimentSpec`: Bundles model, hardware, environment, precision, runtime, UTC timestamp.
  - `BenchmarkResult`: Explicit `ExecutionStatus`, nullable metric fields (`ttft_ms`, `itl_ms_mean`, `itl_ms_p50`, `itl_ms_p90`, `itl_ms_p99`, `throughput_tokens_per_sec`, `peak_vram_used_mb`), and optional `raw_latencies_ms`.
  - `ArtifactManifest`: Canonical schema for the generated artifact with checksum.
- **Verification**: Unit tests validating serialization/deserialization, nullable metrics in diagnostic runs, round-trip JSON, and schema validation.

### Phase 3: Hardware & Environment Detection
- [ ] Implement `rocmhub/hardware/detector.py`:
  - Real detector: inspects `torch.version.hip`, `torch.cuda.get_device_name()`, `/sys/class/kfd`, `rocm-smi` / `rocminfo` for dynamic gfx target string (e.g. `"gfx942"`, `"gfx1100"`, `"unknown"`).
  - Diagnostic/Mock detector: activated when running in non-ROCm environments (e.g. macOS development, CI without AMD GPU), allowing pipeline orchestration and validation without fake metrics.
- [ ] Implement `rocmhub/hardware/environment.py`:
  - Captures driver versions, Linux kernel, ROCm paths, and override variables (`HSA_OVERRIDE_GFX_VERSION`).
- **Verification**: Unit tests on mock detector + real detection test asserting correct field structure.

### Phase 4: Model Source & Inspection
- [ ] Implement `rocmhub/models/source.py`:
  - Resolves model from Hugging Face Hub.
  - Caches files and resolves immutable commit hash (commit SHA).
- [ ] Implement `rocmhub/models/inspector.py`:
  - Parses `config.json` without loading weights into RAM/VRAM.
  - Calculates parameter count from `model.safetensors.index.json` or single `safetensors` headers.
  - Identifies attention type (GQA/MHA), vocabulary size, hidden size, context length.
- **Verification**: Test model inspector against `Qwen/Qwen2.5-0.5B-Instruct` configuration.

### Phase 5: Baseline Runner Adapter
- [ ] Implement `rocmhub/runners/base.py`:
  - Clean abstract protocol: `initialize()`, `warmup()`, `generate_stream()`, `shutdown()`.
- [ ] Implement `rocmhub/runners/hf_runner.py`:
  - Loads model using `AutoModelForCausalLM` and `AutoTokenizer`.
  - Sets precision (`float16` or `bfloat16`).
  - Dispatches to target AMD device (`cuda:0` under HIP).
  - Implements a token-by-token streaming generator to measure exact token arrival times.
- **Verification**: Execute test generation with prompt, asserting valid non-empty tokens and correct device placement.

### Phase 6: Benchmark Harness & Metrics
- [ ] Implement `rocmhub/benchmarks/metrics.py`:
  - Accurate time collection for:
    - Prompt processing start $t_0$.
    - First token arrival $t_1 \rightarrow \text{TTFT} = (t_1 - t_0) \times 1000$ ms.
    - Subsequent tokens $t_i \rightarrow \text{ITL}_i = (t_i - t_{i-1}) \times 1000$ ms.
  - Computes ITL statistics: mean, median (p50), 90th percentile (p90), 99th percentile (p99).
  - Throughput: $\frac{\text{total tokens}}{\sum \text{ITL}}$.
  - Retains raw latency samples (`raw_latencies_ms`).
- [ ] Implement `rocmhub/benchmarks/memory.py`:
  - Queries `torch.cuda.max_memory_allocated()` and tracks peak memory in MB.
- [ ] Implement `rocmhub/benchmarks/harness.py`:
  - Executes configurable warmup iterations (to allow HIP/MIOpen kernel compilation).
  - Executes timed runs and aggregates metric distributions.
  - In diagnostic mode, marks metrics as `NOT_MEASURED` (never synthesizes mock numbers).
- **Verification**: Benchmark metric calculations verified with unit tests.

### Phase 7: Artifact Builder & Manifest Generator
- [ ] Implement `rocmhub/artifacts/builder.py`:
  - Creates artifact directory: `artifacts/<model_slug>_<timestamp>/`.
  - Writes:
    - `manifest.json`: Full specification matching `ArtifactManifest`.
    - `metrics.json`: Detailed percentile latency and memory data.
    - `environment.json`: Host and ROCm specs.
    - `reproduce.sh`: Executable script with exact command and environment variables.
  - Computes manifest SHA256 checksum for tamper evidence.
- **Verification**: Validates created directory layout and verifies manifest JSON matches schema.

### Phase 8: CLI Entrypoint & End-to-End Slice
- [ ] Implement `rocmhub/cli/main.py`:
  - Commands:
    - `rocmhub env`: Displays detected AMD hardware and ROCm status.
    - `rocmhub inspect <model_id>`: Displays model metadata, parameter count, and tensor info.
    - `rocmhub run --model <model_id> [--precision fp16|bf16] [--device 0]`: Runs full pipeline.
  - Clean formatted output table using terminal colors/formatting.
- **Verification**: Run `rocmhub run --model Qwen/Qwen2.5-0.5B-Instruct` end-to-end.

---

## 3. Definition of Done (Criteria of Readiness for Vertical Slice 1)

Vertical Slice 1 is considered complete when all of the following criteria are met:

1. **Deterministic Execution**:
   Running `rocmhub run --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16` completes without uncaught errors from start to finish.
2. **Accurate Hardware & Environment Provenance**:
   The output manifest records the exact GPU name, open `gfx` target string (or diagnostic indicator if running on dev CPU), ROCm/HIP version, PyTorch version, and OS.
3. **Valid Performance Metrics & Status Integrity**:
   - For real GPU runs: reports real, measured TTFT (ms), ITL percentiles (mean, p50, p90, p99 ms), throughput (tokens/s), peak VRAM (MB) with status `SUCCESS`.
   - For diagnostic/dry runs on non-ROCm hosts: status is `NOT_MEASURED`, metrics are `None`, and zero fake performance numbers are generated.
4. **Reproducible Artifact**:
   An output directory is generated containing:
   - `manifest.json` with immutable Git commit SHA, valid SHA256 checksum and schema compliance.
   - `metrics.json`.
   - `reproduce.sh` which can be re-run to reproduce the experiment.
5. **Architectural Cleanliness**:
   - Zero hardcoded references to specific runtime engines in the core domain.
   - Decoupled modules (Model, Hardware, Runner, Benchmark, Artifact).
   - Automated tests (`pytest`) covering data models, inspector, benchmark math, and artifact generation.
6. **Graceful Degradation**:
   On non-AMD machines (e.g. macOS dev environment or CPU CI), the tool does not crash; it reports hardware diagnostics and can execute in a verified diagnostic mode without generating fake performance data.

---

## 4. Next Step

**Phase 1 & 2 Implementation**:
1. Initialize repository structure and configuration (`pyproject.toml`, `.gitignore`, `README.md`).
2. Implement `rocmhub.core.types` with data models (`ModelSpec`, `HardwareSpec`, `EnvironmentSpec`, `ExperimentSpec`, `BenchmarkResult`, `ArtifactManifest`).
3. Add initial unit tests to establish test-driven validation from commit #1.
