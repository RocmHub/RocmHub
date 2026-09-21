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

### Phase 1: Project Foundation & Tooling (COMPLETED)
- [x] Create repository base: `.gitignore`, `pyproject.toml`, `README.md`.
- [x] Define minimal dependencies:
  - `torch` (with ROCm support on AMD systems, standard on dev/CI).
  - `transformers`, `huggingface_hub`, `safetensors`.
  - `pydantic` (for robust schemas, serialization, and validation).
  - `rich` / `typer` or standard `argparse` for clean CLI output.
  - `pytest` for automated testing.
- **Verification**: `pytest` runs and passes with basic smoke test.

### Phase 2: Core Domain Types & Schemas (COMPLETED)
- [x] Implement `rocmhub/core/types.py`:
  - `ExecutionStatus`: Enum (`SUCCESS`, `FAILED`, `SKIPPED`, `NOT_MEASURED`).
  - `ModelSpec`: `schema_version`, model identifier, requested revision, immutable resolved `commit_sha`, architecture, parameter count.
  - `HardwareSpec`: `schema_version`, `gpu_present`, `gpu_vendor`, `device_id`, `device_name`, `family`, open `gfx_target: str | None`, `vram_total_mb`, `vram_free_mb`, `compute_units`, `bus_id`.
  - `EnvironmentSpec`: `schema_version`, OS, kernel, architecture, `rocm_version` (nullable), `hip_version` (nullable), PyTorch version, active flags.
  - `DetectionReport`: Bundles `EnvironmentSpec`, `List[HardwareSpec]`, observation provenance, and diagnostic warnings.
  - `ExperimentSpec`: Bundles model, hardware, environment, precision, runtime, UTC timestamp.
  - `BenchmarkResult`: Explicit `ExecutionStatus`, nullable metric fields (`ttft_ms`, `itl_ms_mean`, `itl_ms_p50`, `itl_ms_p90`, `itl_ms_p99`, `throughput_tokens_per_sec`, `peak_vram_used_mb`), and optional `raw_latencies_ms`.
  - `ArtifactManifest`: Canonical schema for the generated artifact with checksum.
- **Verification**: Unit tests validating serialization/deserialization, nullable metrics in diagnostic runs, round-trip JSON, and schema validation.

### Phase 3: Hardware & Environment Detection (COMPLETED)
- [x] Implement `rocmhub/hardware/base.py`:
  - `HardwareDetector` protocol (`detect_gpus() -> Tuple[List[HardwareSpec], Dict[str, str], List[str]]`).
- [x] Implement `rocmhub/hardware/detector.py`:
  - Strict observation layer: reports actual detected state without policy decisions or compatibility judgments.
  - Multi-tier prioritized fallback:
    1. PyTorch HIP runtime (`torch.cuda` with ROCm)
    2. `rocminfo` (HSA agent enumeration)
    3. `amd-smi` (modern AMD SMI)
    4. `rocm-smi` (legacy SMI)
    5. Linux sysfs / KFD topology (`/sys/class/kfd/topology/nodes/`)
  - Dynamic discovery of `gfx_target` as an open string via regex (e.g. `gfx1100`, `gfx942`, unlisted custom targets).
  - Multi-GPU enumeration support (`List[HardwareSpec]`).
  - Safe subprocess execution: fixed timeout, `LC_ALL=C`, no shell, no sudo, non-zero exits captured as warnings.
  - Diagnostic mode on non-AMD / macOS / CI: returns 0 GPUs cleanly without error (exit code 0, no synthetic metrics).
- [x] Implement `rocmhub/hardware/environment.py`:
  - Captures OS, kernel, architecture, Python, PyTorch/HIP versions, and ROCm paths.
  - Enforces strict whitelist of ROCm environment variables (`HSA_OVERRIDE_GFX_VERSION`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, etc.), preventing sensitive credential or secret leaks.
- [x] CLI command: `rocmhub env [--json]`.
- **Verification**: 60 unit & integration tests covering diagnostic mode, mocked Radeon RX 7900 XTX, Instinct MI300X, multi-GPU discovery, fallback tiers, and secret isolation.

### Phase 4: Model Source & Inspection (COMPLETED)
- [x] Implement `rocmhub/models/base.py`:
  - `ModelSource` protocol (`source_name`, `resolve_revision`, `get_repository_metadata`, `fetch_metadata_file`).
  - `RepositoryMetadata` dataclass.
- [x] Implement `rocmhub/models/huggingface.py`:
  - `HuggingFaceModelSource` adapter using `huggingface_hub` without executing remote code.
  - Resolves revisions to immutable 40-character Git commit SHAs.
  - Translates Hub-specific exceptions on the boundary (`ModelNotFoundError`, `AuthRequiredError`, `RevisionNotFoundError`, `NetworkError`).
- [x] Implement `rocmhub/models/inspector.py`:
  - Statically parses `config.json` without downloading or loading weight tensors.
  - Safe parameter count discovery from Hub-level safetensors metadata / index (`None` if indeterminable).
  - Extracts architecture, context length, default precision (`torch_dtype`), and weights format (`safetensors`, `pytorch_bin`, `gguf`, `unknown`).
  - Enforces `trust_remote_code=False` with `RemoteCodeRequiredError`.
- [x] CLI command: `rocmhub inspect <model_id> [--revision <rev>] [--json]`.
- **Verification**: Complete offline unit test suite (`tests/test_models.py`, `tests/test_cli.py`) + live network integration test (`tests/test_integration_hf.py`).

### Phase 5: Capability / Compatibility Evaluation (COMPLETED)
- [x] Implement `rocmhub/capabilities/policy.py`:
  - `CapabilityPolicy` versioned representation separating host observations from interpretation rules.
  - Known supported architecture sets (CDNA, RDNA) without hardcoding closed enums.
- [x] Implement `rocmhub/capabilities/evaluator.py`:
  - `CapabilityEvaluator` linking `ModelSpec` and `DetectionReport` into `CapabilityReport`.
  - Non-binary verdicts: `READY`, `BLOCKED`, `UNKNOWN`, `NO_ACCELERATOR`.
  - UNKNOWN never automatically becomes BLOCKED (ensuring unlisted/new AMD GPUs remain UNKNOWN).
  - Multi-GPU individual device evaluation (`DeviceCapabilityAssessment`).
  - Conservative capabilities: `amd_gpu_present`, `rocm_detected`, `hip_detected`, `torch_available`, `torch_hip_available`, `model_metadata_complete`, `remote_code_required`, `baseline_runtime_candidate`.
  - Zero weight downloads, zero inference executions.
- [x] CLI command: `rocmhub check <model_id> [--revision <rev>] [--json]`:
  - Human-readable summary table and pure JSON output.
  - Standardized exit codes: 0 (READY), 2 (NO_ACCELERATOR), 3 (BLOCKED), 4 (UNKNOWN), 1 (Error).
- **Verification**: 79 tests (unit tests covering READY, NO_ACCELERATOR, ROCm missing, CPU PyTorch, unknown GFX, incomplete metadata, multi-GPU, structured reason codes + live network integration test on macOS).

### Phase 5: Baseline Runner Adapter
- [x] Implement `rocmhub/runners/base.py`:
  - Abstract `BaseRunner` Protocol: `supports()`, `load()`, `generate()`, `unload()`.
- [x] Implement `rocmhub/runners/hf_runner.py` (`pytorch_transformers_hip`):
  - Model loading with immutable `revision=model.commit_sha` and `trust_remote_code=False`.
  - Explicit device binding (`cuda:<device_id>`), no `device_map="auto"`.
  - Strict precision validation (`fp32`, `fp16`, `bf16`).
  - Deterministic generation (`temperature=0.0`, `do_sample=False`).
  - Strict token accounting: `generated_tokens` counts only newly generated tokens, excluding prompt.
  - Resource cleanup on `unload()`: `gc.collect()` and `torch.cuda.empty_cache()`.
  - Resilient lazy torch/transformers imports with dependency injection support for testability.
- [x] Add CLI command `rocmhub run <model_id> [--revision <rev>] [--device <id>] [--precision <dtype>] [--prompt <text>] [--max-new-tokens <n>] [--json]`:
  - Preflight evaluation gating: halts immediately without downloading weights or running inference if `verdict != READY` (exits with 2 for `NO_ACCELERATOR`, 3 for `BLOCKED`, 4 for `UNKNOWN`).
  - Outputs human-readable summary or structured `RunResult` JSON.
- **Verification**: 96 tests (93 offline unit tests covering lifecycle, token counts, error states, preflight gating + 3 live network integration tests).

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
