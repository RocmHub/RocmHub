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

### Phase 5: Baseline Runner Adapter (COMPLETED)
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

### Phase 6: Benchmark Harness & Metrics (COMPLETED)
- [x] Implement `rocmhub/benchmarks/base.py`:
  - `BenchmarkConfig` with bounds validation (`warmup_runs >= 0`, `measurement_runs >= 1`, `max_new_tokens > 0`, `device_id >= 0`).
  - `BenchmarkRunMeasurement` structure storing raw timing evidence per run.
- [x] Implement `rocmhub/benchmarks/streaming.py`:
  - `TokenTimestampStreamer` recording high-resolution monotonic timestamps (`time.perf_counter_ns`) upon token emission.
- [x] Implement `rocmhub/benchmarks/memory.py`:
  - `MemoryTracker` querying PyTorch allocator peak memory stats (`torch.cuda.max_memory_allocated`).
- [x] Implement `rocmhub/benchmarks/metrics.py`:
  - Exact TTFT: $(t_{\text{first\_token}} - t_{\text{request\_start}})$ in ms.
  - Consecutive ITL deltas $(t_{i+1} - t_i)$ in ms, strictly excluding TTFT.
  - End-to-end throughput: $\frac{\text{generated\_tokens}}{\text{total\_generation\_time\_seconds}}$.
  - Deterministic linear-interpolation percentiles (p50, p90, p99) without external dependencies.
  - Warmup runs strictly excluded from performance summary metrics.
  - Partial failure rule: if any measurement run fails, entire benchmark is marked `FAILED` with `None` performance metrics.
- [x] Implement `rocmhub/benchmarks/harness.py`:
  - Coordinates warmup, measurement iterations, and accelerator device synchronization (`torch.cuda.synchronize`).
- [x] Implement CLI command `rocmhub benchmark <model_id>`:
  - Preflight gating: halts with codes 2 (`NO_ACCELERATOR`), 3 (`BLOCKED`), 4 (`UNKNOWN`) without downloading weights.
  - Clean human-readable table and JSON outputs.
- **Verification**: 117 tests (113 offline unit tests covering TTFT, ITL, percentiles, memory, sync, failure states, preflight gates + 4 live network integration tests).

### Phase 7: Correctness & Quality Validation (COMPLETED)
**Principal: БЫСТРЕЕ ≠ ЛУЧШЕ, ЕСЛИ МОДЕЛЬ СТАЛА ХУЖЕ.**
- [x] Implement `rocmhub/validation/base.py`:
  - `ValidationCase`: stable `case_id`, `prompt`, `max_new_tokens`, `critical`, `expected_pattern`.
  - `ValidationConfig`: immutable Pydantic model with bounds-validated `max_new_tokens`, `quality_threshold`, `precision`, `device_id`.
  - `DEFAULT_VALIDATION_CASES`: 3 stable cases (`basic_completion_001`, `instruction_following_001`, `deterministic_generation_001`).
- [x] Implement `rocmhub/validation/correctness.py`:
  - `CorrectnessEvaluator`: inference completion, non-empty output, positive token count, NaN/Inf/`\ufffd` corruption detection, regex pattern matching, determinism (repeated runs under fixed seed).
  - `CorrectnessResult`: `passed`, `checks_run`, `checks_passed`, `checks_failed`, `failures`.
- [x] Implement `rocmhub/validation/quality.py`:
  - `QualityMetric` Protocol: composable per-case scoring interface.
  - `ExactTokenAgreementMetric`: character-exact agreement (1.0 or 0.0).
  - `NormalizedTextAgreementMetric`: Jaccard word-overlap ratio.
  - `QualityEvaluator`: self-validation mode → `quality_measured=False`, `qrr_percent=None` strictly.
  - Comparison mode → `QRR = (candidate_score / baseline_score) * 100`; division-by-zero → `None`.
- [x] Implement `rocmhub/validation/evaluator.py`:
  - `ValidationEvaluator.run_self_validation()`: executes suite on baseline runner; correctness gating; `quality_measured=False`, `qrr_percent=None` always.
  - `ValidationEvaluator.run_comparison()`: executes suite on both runners; QRR-gated verdict.
  - Verdict assignment: `PASS | FAIL | INCONCLUSIVE | NOT_MEASURED`.
- [x] Add `ValidationCase`, `ValidationRunResult`, `ValidationReport` to `core/types.py`:
  - `ValidationRunResult` Pydantic validator: SUCCESS requires non-null `generated_text` and `generated_tokens >= 0`; SKIPPED/NOT_MEASURED forbid token fields.
  - `ValidationReport` Pydantic validator: PASS requires `correctness_passed=True` and `critical_cases_failed=0`; NOT_MEASURED requires all metrics `None`.
- [x] Add validation errors to `core/errors.py`:
  - `ValidationError`, `InvalidValidationConfigError`, `CorrectnessGateFailedError`, `QualityGateFailedError`.
- [x] Implement CLI command `rocmhub validate <model_id> [--revision] [--device] [--precision] [--max-new-tokens] [--json]`:
  - Preflight gate identical to `run`/`benchmark`: exits 2 (NO_ACCELERATOR), 3 (BLOCKED), 4 (UNKNOWN) without downloading weights.
  - On preflight failure: constructs `ValidationReport(verdict=NOT_MEASURED)` with all metrics `None`.
  - On READY: loads runner, runs `ValidationEvaluator.run_self_validation()`, unloads runner in `finally`.
  - Exit codes: `0=PASS`, `2=NOT_MEASURED`, `3=FAIL`, `4=INCONCLUSIVE`, `1=error`.
- **Validation Independence**: `ValidationEvaluator` is completely decoupled from `BenchmarkHarness` — no shared state, no combined scores.
- **Verification**: 201 offline tests (88 new validation tests + 113 regression-free existing tests) + 5 live network integration tests.

### Phase 8: Reproducible Artifact Builder (COMPLETED)
- [x] Implement `rocmhub/artifacts/integrity.py`:
  - Canonical JSON serialization (`sort_keys=True`, compact stable separators `","`, `":"`, strict `allow_nan=False`, UTF-8).
  - Cryptographic hashing: `compute_sha256()` and `compute_file_sha256()`.
  - Fail-closed secret scanner `scan_for_secrets()` rejecting credentials, passwords, private keys, and tokens while explicitly whitelisting legitimate token counts (`generated_tokens`, `input_tokens`, `max_new_tokens`, `tokens_per_sec`).
- [x] Implement `rocmhub/artifacts/manifest.py`:
  - Deterministic `experiment_id`: `exp-<sha256[:24]>` derived from input experiment configuration (model, immutable commit SHA, hardware, precision, benchmark params, validation config).
  - Deterministic `artifact_id`: `art-<sha256[:24]>` derived from `experiment_id` + canonical payload file hashes.
  - Declarative `ReproductionMetadata` (reproduce.json).
  - `build_manifest()`: Top-level `ArtifactManifest` recording file inventory and detached checksums.
- [x] Implement `rocmhub/artifacts/storage.py`:
  - `LocalArtifactStore`: Atomic directory commit (POSIX rename/replace on same filesystem).
  - Prevention of silent overwrite via `ArtifactConflictError`.
  - Comprehensive tamper verification `verify_artifact()`: checks schema, inventory, missing files, modified files, and unexpected files.
- [x] Implement `rocmhub/artifacts/builder.py`:
  - `ArtifactBuilder`: Orchestrates staging, secret scanning, canonical file writes, per-file hashing, manifest assembly, detached checksum generation, and atomic publication.
  - Separate raw benchmark data storage (`benchmark_raw.json`).
  - Automatic cleanup of staging directory on failure (leaves no corrupted final artifact).
  - Faithful diagnostic artifact mode: marked `COMPLETE` with preserved `NO_ACCELERATOR`, `SKIPPED`, `NOT_MEASURED` without downloading model weights.
- [x] Implement CLI commands:
  - `rocmhub artifact build <model_id> [--revision] [--device] [--precision] [--output-dir] [--json]`
  - `rocmhub artifact verify <path> [--json]`
- **Core Invariant**: $\text{Artifact COMPLETE} \ne \text{Execution SUCCESS} \ne \text{Validation PASS} \ne \text{ROCmHub Verified}$.
- **Verification**: 228 offline unit tests + 6 live network integration tests. Ruff clean. Mypy clean.

### Phase 9: Benchmark Guard & Reproducibility Gate (COMPLETED)
**Principle: BenchmarkHarness измеряет. BenchmarkGuard проверяет условия измерения и воспроизводимость.**
- [x] Implement `rocmhub/guard/base.py`:
  - Standard reason codes: `NO_BENCHMARK_EXECUTION`, `ARTIFACT_INTEGRITY_FAILED`, `INSUFFICIENT_MEASUREMENT_RUNS`, `EVIDENCE_INCONSISTENT`, `SUMMARY_MISMATCH`, `ENVIRONMENT_DRIFT`, `HARDWARE_THROTTLING_DETECTED`, `HARDWARE_ECC_ERRORS`, `HIGH_VARIABILITY`, `STABLE_MEASUREMENT`, `REFERENCE_DRIFT`, `TELEMETRY_UNAVAILABLE`.
  - Default `DEFAULT_GUARD_POLICY` (min 5 runs, 15% TTFT MAD, 10% throughput MAD, 15% latency MAD, summary recomputation tolerance 1e-3).
  - `ABBASequence` data contract for interleaved A/B/B/A baseline/candidate execution cycles.
- [x] Implement `rocmhub/guard/statistics.py`:
  - Sample median `compute_median(values)`.
  - Median Absolute Deviation `compute_mad(values)`: $\text{MAD} = \text{median}(|x_i - \text{median}(X)|)$.
  - Relative MAD `compute_relative_mad(values)`: $\frac{\text{MAD}}{\text{median}}$.
  - Independent headline summary recomputation `recompute_benchmark_summary(measurements)`.
  - `verify_summary_against_raw()`: detects discrepancies and tampering between raw run timestamps and reported metrics.
- [x] Implement `rocmhub/guard/environment.py`:
  - `extract_environment_fingerprint()`: extracts normalized deterministic dictionary, strips ephemeral noise, computes canonical SHA-256 hash.
  - `compare_fingerprints()`: detects exact drift and returns diff dictionary.
- [x] Implement `rocmhub/guard/reference.py`:
  - `evaluate_hardware_health()`: validates throttling and uncorrectable ECC errors.
  - `compare_reference_measurements()`: evaluates calibration drift; returns `None` if reference was not measured (never `True`).
- [x] Implement `rocmhub/guard/evaluator.py`:
  - `BenchmarkGuard.evaluate(artifact_path)`: multi-gate verification pipeline (Integrity -> Execution Check -> Environment Match -> Hardware Health -> Raw Evidence Completeness -> Summary Recomputation -> MAD Dispersion -> Reference Calibration).
  - Diagnostic artifacts cleanly evaluate to `NOT_MEASURED` (reason: `NO_BENCHMARK_EXECUTION`).
  - Corrupted/tampered artifacts fail immediately (`ARTIFACT_INTEGRITY_FAILED`).
  - Verdict system: `PASS | FAIL | INCONCLUSIVE | NOT_MEASURED`.
- [x] Implement CLI command `rocmhub guard <path> [--json]`:
  - Formatted human-readable summary table and structured JSON report.
  - Exit codes: 0 (PASS), 2 (NOT_MEASURED), 3 (FAIL), 4 (INCONCLUSIVE), 1 (Error).
- **Verification**: 244 offline tests (16 new Phase 9 tests) + 7 live network integration tests. Ruff clean. Mypy clean. Real CLI verification on macOS returns `NOT_MEASURED` with code 2.

### Phase 10: Model Forge Foundation (COMPLETED)
**Objective: Automated preparation and compilation of open-source models for AMD GPUs.**
- [x] Implement `src/rocmhub/forge/base.py`:
  - `BuildStatus`: Enum (`PREPARED`, `EXECUTED`, `FAILED`).
  - `StepStatus`: Enum (`PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, `SKIPPED`).
  - `BuildStepSpec`, `BuildStepRecord`, `MaterializedModel`.
  - `ForgePlan`: deterministic hash, model ID, immutable 40-char SHA, precision, target GPU, recipe, steps, estimated disk bytes.
- [x] Implement `src/rocmhub/forge/recipes.py`:
  - `ForgeRecipe` protocol and `PyTorchTransformersHipRecipe` (v1.0.0, `pytorch_transformers_hip`).
  - Causal language model matching (`Qwen2ForCausalLM`, `LlamaForCausalLM`, etc.).
  - Supported precisions (`fp16`, `bf16`, `fp32`).
  - Generates `runtime_config.json`, `model_config.json`, `recipe.json`, and executable `run_inference.py`.
- [x] Implement `src/rocmhub/forge/planner.py`:
  - `ForgePlanner.create_plan()`: resolves immutable 40-char commit SHA via `ModelInspector` without downloading weight tensors.
  - Matches recipe, validates architecture and precision.
  - Detects target GPU from `SystemObserver` and confirms compatibility via `CapabilityEvaluator`.
  - Computes required disk space with safety margin.
- [x] Implement `src/rocmhub/forge/materializer.py`:
  - `ModelMaterializer`: preflight disk space checks (`InsufficientDiskSpaceError`).
  - Enforces `trust_remote_code=False` and uses immutable commit SHA.
  - Handles gated repo (`AuthRequiredError`) and missing repo (`ModelNotFoundError`).
  - Supports `--no-weights` for fast metadata-only materialization.
- [x] Implement `src/rocmhub/forge/manifest.py`:
  - `BuildManifest` schema recording build ID, plan ID, status, runtime, steps, and artifact checksums.
  - Secret sanitization: redacting tokens and API keys, asserting zero leaked credentials.
  - Atomic, canonical JSON writer (`write_manifest()`).
- [x] Implement `src/rocmhub/forge/executor.py`:
  - `ForgeExecutor.execute()`: executes build steps (`check_prerequisites`, `materialize_model`, `configure_runtime`, `generate_launch_scripts`, `verify_build`).
  - Manages build directory, detects conflicts without `--force` (`BuildConflictError`).
  - Strict semantics: `--no-weights` sets `BuildStatus.CONFIG_ONLY` (inference strictly forbidden on `CONFIG_ONLY`).
  - `PREPARED` requires weight shards, tokenizer, and configs verified present on disk.
  - Non-AMD behavior: marks status `PREPARED` or `CONFIG_ONLY`, `amd_validated = False`. Real execution step cleanly skipped without fake AMD inference.
- [x] CLI commands:
  - `rocmhub forge plan <model_id> [--revision] [--precision] [--target-gpu] [--output-dir] [--recipe] [--json]`
  - `rocmhub forge build <model_id> [--revision] [--precision] [--target-gpu] [--output-dir] [--recipe] [--force] [--no-weights] [--execute] [--json]`
- **Verification**: 272 offline unit tests + 8 live network integration tests (all passing). Ruff clean. Mypy clean.

### Phase 11: Autonomous AI Engineer MVP (COMPLETED)
**Objective: Specialized AI Engineer autonomously preparing open-source models for AMD GPUs.**
- [x] Implement `src/rocmhub/engineer/base.py`:
  - Data contracts: `EngineerObjective` (`PREPARE_AMD`, `BASELINE_RUN`, `BENCHMARK_AMD`), `EngineerStatus` (`RUNNING`, `SUCCESS`, `STOPPED_ENVIRONMENT`, `BUDGET_EXCEEDED`, `FAILED`).
  - `EngineerBudget`: `max_attempts` (default 5), `max_execution_time_seconds` (default 600), `max_disk_usage_bytes` (default 10 GB).
  - `TrajectoryStep`: immutable step recording step index, phase (`OBSERVE`, `PLAN`, `ACT`, `EVALUATE`, `REVISE`), tool call, observation, and status.
  - `EngineerRequest` and `EngineerReport`.
- [x] Implement `src/rocmhub/engineer/provider.py`:
  - `LLMProvider` Protocol.
  - `AutonomousRulesProvider`: 100% deterministic offline expert decision tree driving the autonomous loop reliably without network/LLM dependencies.
  - `OpenAICompatibleProvider`: provider reading `ROCMHUB_LLM_*` env vars, zero credential leaks, schema-enforced actions.
- [x] Implement `src/rocmhub/engineer/tools.py`:
  - Restricted Tool Registry with 10 tools: `inspect_model`, `inspect_hardware`, `check_capability`, `create_forge_plan`, `materialize_model`, `execute_forge_build`, `run_baseline`, `run_benchmark`, `read_build_errors`, `save_engineer_report`.
  - Security boundary: strict path traversal validation (`validate_safe_path`), forbidden root prefixes (`FORBIDDEN_PREFIXES`), prompt injection sanitization (`sanitize_untrusted_text`).
  - No arbitrary shell, Python eval, or system command execution.
- [x] Implement `src/rocmhub/engineer/policy.py`:
  - `BudgetGuard`: tracks execution time, step attempts, disk usage against limits.
  - `LoopDetector`: tracks per-action failure counts and detects repeating/oscillating patterns (prevents ping-pong loops).
  - `FailureClassifier`: categorizes errors into `HARDWARE_MISMATCH`, `DOWNLOAD_FAILURE`, `OUT_OF_MEMORY`, `RECIPE_INCOMPATIBLE`, `ROCM_RUNTIME_ERROR`, `UNKNOWN_FAILURE`.
- [x] Implement `src/rocmhub/engineer/memory.py`:
  - `TrajectoryStore`: logs steps to local `trajectory.jsonl` and final `report.json`.
  - Fail-closed secret scrubber redacting API keys, Bearer tokens, and secrets from trajectories.
- [x] Implement `src/rocmhub/engineer/agent.py`:
  - `AIEngineer`: autonomous control loop `OBSERVE -> PLAN -> ACT -> EVALUATE -> REVISE`.
  - Deterministic execution guard: LLM cannot mark `SUCCESS` or `AMD_VALIDATED` directly; only real verified executor outputs dictate status.
  - Non-AMD behavior: cleanly handles non-AMD environment by stopping gracefully at `STOPPED_ENVIRONMENT` or completing model preparation cleanly with zero synthetic GPU metrics.
- [x] Implement CLI command:
  - `rocmhub engineer <model_id> [--revision] [--target-gpu] [--objective] [--max-attempts] [--max-minutes] [--max-disk-gb] [--allow-full-weights] [--output-dir] [--json]`
- **Verification**: 306 unit tests (24 new Phase 11 tests) + 9 live integration tests passing. Ruff clean. Mypy clean. Real CLI execution on macOS verified.

### Phase 12: Optimization Engine (COMPLETED)
**Objective: Systematic candidate generation, building, execution, and objective comparison against immutable baselines on AMD GPUs.**
- [x] Harden success semantics across AI Engineer and Forge:
  - Added `EngineerStatus.CONFIG_ONLY` to prevent conflating configuration preparation with full inference execution or verified optimization.
- [x] Implement domain exceptions in `src/rocmhub/core/errors.py`:
  - `OptimizationError`, `UnsupportedStrategyError`, `BaselineExecutionError`, `CandidateBuildError`, `IncomparableResultsError`, `QualityRegressionError`.
- [x] Implement `src/rocmhub/optimization/base.py`:
  - `OptimizationStrategy`: `BF16`, `FP16`, `FP32`, `TORCH_COMPILE`, `QUANT_INT8`, `QUANT_FP8`, `CUSTOM`.
  - `CandidateStatus`: `PLANNED`, `CONFIG_ONLY`, `PREPARED`, `EXECUTED`, `FAILED`, `UNSUPPORTED`.
  - `ComparisonVerdict`: `IMPROVED`, `REGRESSED`, `NO_CHANGE`, `NOT_MEASURED`, `INCOMPARABLE`.
  - Data contracts: `OptimizationCandidate`, `OptimizationBaseline`, `ComparisonResult`, `OptimizationRequest`, `OptimizationPlan`, `OptimizationReport`.
- [x] Implement `src/rocmhub/optimization/recipes.py`:
  - `OptimizationRecipe` protocol: `strategy`, `name`, `description`, `is_supported(hardware, env) -> (bool, reason)`, `apply(forge_recipe, model_spec) -> forge_recipe`.
  - Concrete strategies: `BF16OptimizationRecipe`, `FP16OptimizationRecipe`, `FP32OptimizationRecipe`, `TorchCompileRecipe`.
  - Strict quantization safety: `QuantizationRecipe` (INT8/FP8) checks for explicit backend libraries (`bitsandbytes`, `autoawq`, AMD FP8 kernels on CDNA3 gfx942). If absent, explicitly flags `is_supported = False` with `UNSUPPORTED` reason. Zero fake quantization.
- [x] Implement `src/rocmhub/optimization/baseline.py`:
  - `BaselineManager`: establishes immutable, reproducible reference baseline with fixed model ID, 40-char SHA, hardware, precision, and runtime.
  - Non-AMD behavior: marks baseline as `CONFIG_ONLY` or `PREPARED`, with performance metrics `NOT_MEASURED` (zero fake GPU metrics).
- [x] Implement `src/rocmhub/optimization/comparison.py`:
  - `ComparisonEngine`: strict comparability assertions (matching model ID, commit SHA, hardware target).
  - Truthful speedup calculations: speedup is computed ONLY when real baseline and candidate benchmark results exist; otherwise returns `ComparisonVerdict.NOT_MEASURED`.
  - Quality retention: evaluates Quality Retention Rate (`qrr_percent`). If quality degrades below threshold (e.g. 95%), candidate is marked `REGRESSED`.
  - Statistical significance: requires >3% delta to declare `IMPROVED` or `REGRESSED`; otherwise `NO_CHANGE`.
- [x] Implement `src/rocmhub/optimization/executor.py`:
  - `OptimizationExecutor`: orchestrates baseline establishment, candidate planning, Forge building, AMD GPU check (stops safely before GPU execution on non-AMD environments), comparative evaluation, and reporting.
- [x] Implement `src/rocmhub/optimization/reports.py`:
  - `format_optimization_report_table()`: Rich terminal comparison matrix.
- [x] AI Engineer integration in `src/rocmhub/engineer/tools.py`:
  - Added 6 safe tools to `ToolRegistry`: `create_optimization_plan`, `build_candidate`, `execute_candidate`, `benchmark_candidate`, `compare_candidates`, `read_optimization_errors`.
  - Enforced path traversal validation (`validate_safe_path`) and forbidden prefixes across all optimization tools.
- [x] Implement CLI command:
  - `rocmhub optimize <model_id> [--revision] [--target-gpu] [--objective] [--max-candidates] [--max-minutes] [--allow-full-weights] [--output-dir] [--json]`
- **Verification**: 327 unit tests (21 new Phase 12 tests) + 10 live integration tests passing. Ruff clean. Mypy clean. Live CLI execution on macOS verified (exits 0 with status `CONFIG_ONLY`, `NOT_MEASURED`, zero synthetic metrics).

### Phase 13: Backend API & Job Orchestration (COMPLETED)
**Objective: Local FastAPI backend and asynchronous job orchestration engine with SQLite persistence, SSE streaming, and security sandboxing.**
- [x] Integrate FastAPI and Uvicorn into project packaging (`pyproject.toml`).
- [x] Implement server configuration and security layer (`src/rocmhub/server/config.py`, `src/rocmhub/server/security.py`):
  - Localhost binding by default (`127.0.0.1`).
  - Request body size limit middleware (`max_request_bytes=1MB`, HTTP 413).
  - Path traversal and system directory protection (`validate_job_path`).
  - Secret and sensitive token scrubbing (`redact_secrets`, `sanitize_payload`).
- [x] Implement SQLite persistence with versioned schema migrations (`src/rocmhub/server/orchestrator/db.py`, `src/rocmhub/server/orchestrator/migrations.py`):
  - `schema_version`, `jobs`, `job_events` tables with WAL mode, foreign keys, and indexes.
  - Interrupted job recovery on startup (jobs left in `RUNNING` or `QUEUED` marked `FAILED`).
- [x] Implement local Job Manager and background worker (`src/rocmhub/server/orchestrator/manager.py`, `src/rocmhub/server/orchestrator/worker.py`):
  - FIFO bounded queue with directory locking to prevent concurrent writes to the same build folder.
  - Cooperative job cancellation tokens.
  - Strict separation of HTTP job status (`QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`) from domain status (`CONFIG_ONLY`, `PREPARED`, `EXECUTED`, `NOT_MEASURED`).
- [x] Implement Server-Sent Events (SSE) streaming (`src/rocmhub/server/events.py`):
  - Reconnection support via `Last-Event-ID` header and `?from_event_id` query param.
  - Heartbeat keep-alive generator (`: ping\n\n`).
- [x] Implement versioned REST API routes (`src/rocmhub/server/routes/`):
  - `GET /health`: system platform, hardware, and queue status.
  - `GET /api/v1/models/{model_id:path}`: model metadata and host GPU summary.
  - `POST /api/v1/forge/plan`: deterministic build plan generation.
  - `POST /api/v1/jobs`: asynchronous job submission for `FORGE_BUILD`, `ENGINEER`, `OPTIMIZATION`.
  - `GET /api/v1/jobs/{job_id}`: job lifecycle metadata.
  - `GET /api/v1/jobs/{job_id}/events`: SSE progress stream.
  - `GET /api/v1/jobs/{job_id}/result`: structured domain output payload.
  - `POST /api/v1/jobs/{job_id}/cancel`: cooperative job cancellation.
- [x] Implement CLI command:
  - `rocmhub serve [--host 127.0.0.1] [--port 8000] [--db-path PATH]`
- **Verification**: 345 unit tests (18 new Phase 13 tests) + 10 live integration tests passing. Ruff clean. Mypy clean (82 files). Real local HTTP smoke test on `127.0.0.1:8765` for `Qwen/Qwen2.5-0.5B-Instruct` verified (SSE events streamed, terminal status `SUCCEEDED`, domain status `CONFIG_ONLY`, zero synthetic metrics).

### Phase 14: Frontend MVP & API Audit Fixes (COMPLETED)
- [x] Complete backend audit & critical integrity fixes:
  - Commit SHA resolution and persistence in SQLite (`revision` column updated from resolved Forge/Engineer/Optimization manifests).
  - Runtime recipe naming truthfulness (`pytorch_transformers_hip` accurately reflected; no synthetic vLLM claims).
  - Hardware telemetry integrity: Apple Silicon/macOS Darwin host memory strictly decoupled from AMD ROCm GPU VRAM; `rocm_available: false` and explanatory warnings served.
  - Crash recovery error code distinction: `EXECUTION_INTERRUPTED_BY_RESTART` for running jobs vs `QUEUE_DISCARDED_ON_RESTART` for queued jobs.
  - Implemented `GET /api/v1/jobs` pagination (`limit`, `offset`) and filtering (`status`, `job_type`).
- [x] Implement modern responsive Frontend Web Application (`frontend/`):
  - Architecture: React 18, TypeScript, Vite, Tailwind CSS, TanStack Query, Lucide icons.
  - Palette: Dark industrial theme (`#101014` bg, `#19191F` card, `#222229` elevated, `#ED1C24` primary accent).
  - SSE streaming client (`sse.ts`) with automatic reconnect, resume via `from_event_id`, deduplication, and resource cleanup.
  - Semantic status display: explicit domain statuses (`CONFIG_ONLY`, `PREPARED`, `EXECUTED`, `NOT_MEASURED`) separated from HTTP job statuses (`QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`).
  - 5 interactive views:
    - **Dashboard**: Hardware health telemetry, ROCm availability banner, active and historical jobs table with auto-refresh.
    - **Model Explorer**: Hugging Face metadata inspector, commit SHA, architecture, parameters, license, compatibility verdict.
    - **Forge Studio**: Target GPU and precision selection, deterministic build plan inspection, build runner with real-time SSE log viewer and verified artifact digest.
    - **AI Engineer**: Autonomous loop launcher (`BASE_PREPARATION`), budget configuration, live trajectory viewer, cancellation support, executive summary report.
    - **Optimization Lab**: Multi-strategy candidate comparison table, baseline metrics, clear `NOT_MEASURED` tagging on non-ROCm hosts without synthetic numbers.
- **Verification**:
  - Python test suite: 347 unit tests passed (`pytest tests/test_api.py -v` 20/20 passed).
  - Frontend test suite: 12 Vitest tests passed (`npm test`).
  - Production build: `tsc && vite build` built in 1.25s with 0 errors.
  - Code hygiene: `ruff check` and `mypy` clean (82 source files).
  - Live Demo Smoke Test: `scripts/live_demo_smoke.py` end-to-end 10-step automated verification on `Qwen/Qwen2.5-0.5B-Instruct` passed completely.

### Phase 15: Product Acceptance & Polish (COMPLETED)
- [x] Full real-browser end-to-end acceptance testing in Google Chrome across Desktop (1280x860) and Mobile (375x812) viewports.
- [x] Full user scenario verification on `Qwen/Qwen2.5-0.5B-Instruct` across Model Explorer, Forge Studio, AI Engineer, Optimization Lab, and Dashboard.
- [x] Immediate terminal event dispatch and SSE stream lifecycle stabilization.
- [x] Truthful domain status presentation (`CONFIG_ONLY`, `PREPARED`, `EXECUTED`, `NOT_MEASURED`) without synthetic numbers on non-AMD hosts.
- [x] Responsive mobile navigation drawer and cross-studio deep linking.
- [x] 7 verified screenshots captured and archived.
- **Verification**: `scripts/browser_e2e.js` 100% pass, 347 Python unit tests pass, 12 Vitest tests pass.

### Phase 16: CI, Licensing & Repository Hygiene (COMPLETED)
- [x] Standard Apache License 2.0 (`LICENSE`) added in accordance with `pyproject.toml` and `README.md`.
- [x] Multi-version GitHub Actions CI pipeline (`.github/workflows/ci.yml`) covering Python 3.9-3.12 (Ruff, Mypy, Pytest) and Node.js 20 (Vitest, Vite Build).
- [x] Documentation alignment and removal of unverified/false claims across `README.md`, `DEVELOPMENT_PLAN.md`, and `ARCHITECTURE.md`.
- **Verification**: Full test suite pass across backend and frontend, clean linting and typing.

### Phase 17: AMD Execution Readiness (COMPLETED)
- [x] Direct launcher execution validation for materialized models on AMD ROCm hardware (`run_inference.py`).
- [x] Structured JSON output mode (`--json`) added to standalone launcher template with execution metrics (load time, generation time, tokens per second, exit code).
- [x] Safe subprocess execution harness (`ForgeExecutor.execute_build`) with stdout/stderr capture, timeout bounds, and exit code validation.
- [x] Bridge ForgeExecutor `execute_inference` to real launcher execution when ROCm GPU is available.
- [x] Truthful execution status transition: `PREPARED` -> `EXECUTED` upon verified model generation with `amd_validated=True`.
- [x] CLI command: `rocmhub forge execute <build_dir> [--prompt <text>] [--device <id>] [--max-new-tokens <n>] [--timeout <sec>] [--json]`.
- [x] Worker integration: forwarded `execute_inference` flag from orchestrator payload to ForgeExecutor.
- **Verification**: 352 Pytest unit tests passed (+5 new tests in `TestForgeExecutionReadiness`), Ruff & Mypy clean, Vitest 12/12 passed, Vite build clean.

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
   - `metrics.json` / `benchmark.json` and `benchmark_raw.json`.
   - `reproduce.json` which can be used to reproduce the experiment.
5. **Architectural Cleanliness**:
   - Zero hardcoded references to specific runtime engines in the core domain.
   - Decoupled modules (Model, Hardware, Runner, Benchmark, Validation, Artifact, Guard).
   - Automated tests (`pytest`) covering data models, inspector, benchmark math, validation, artifacts, and guard.
6. **Graceful Degradation**:
   On non-AMD machines (e.g. macOS dev environment or CPU CI), the tool does not crash; it reports hardware diagnostics and can execute in a verified diagnostic mode without generating fake performance data.

