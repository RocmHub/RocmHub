# ROCmHub — First AMD Run Guide (`FIRST_AMD_RUN.md`)

This guide provides the complete, reproducible operational procedure for executing and validating open models on AMD ROCm hardware (AMD Radeon RX 7000 series RDNA3 and AMD Instinct MI200/MI300 series CDNA).

---

## 1. Safety Principles & Truth in Execution

ROCmHub enforces strict distinction between process execution and verified AMD GPU acceleration. The status `EXECUTED` and the assertion `amd_validated=True` are **never** awarded based solely on a process exit code of 0 or a text response.

Five independent facts are strictly separated:
1. **Process Success**: Subprocess terminated with exit code 0 without timeout or segfault.
2. **Inference Executed**: Non-empty text output was generated and tokens generated count is strictly positive (\(> 0\)).
3. **HIP Runtime Used**: PyTorch reported active AMD HIP backend (`torch.version.hip` is not null).
4. **AMD GPU Hardware Used**: Execution occurred on a confirmed AMD GPU device (verified via HIP device query; not CPU fallback and not NVIDIA CUDA).
5. **Output Validation Passed**: Generated output passed the technical correctness gate (free of `NaN`, `inf`, UTF-8 replacement characters `\ufffd`, or token count anomalies). Successful inference does **not** imply automatic validation pass.

On non-AMD machines (e.g., Apple Silicon macOS, developer laptops, or CPU-only CI), ROCmHub operates in **diagnostic mode**:
- Model weights are **never** downloaded.
- GPU kernels are **never** executed.
- Performance metrics are reported truthfully as `NOT_MEASURED` (zero synthetic numbers).
- Status remains `CONFIG_ONLY` or `PREPARED`, never `EXECUTED`.

---

## 2. Prerequisites for Real AMD Execution

### Supported Hardware
- **AMD Instinct Series** (CDNA Architecture):
  - MI300X, MI300A (`gfx942`)
  - MI250X, MI250, MI210 (`gfx90a`)
  - MI100 (`gfx908`)
- **AMD Radeon Series** (RDNA3 Architecture):
  - Radeon RX 7900 XTX, 7900 XT, 7900 GRE (`gfx1100`)
  - Radeon PRO W7900, W7800 (`gfx1100`)
  - Radeon RX 7800 XT, 7700 XT (`gfx1101`)

### Software & Environment
- **Operating System**: Linux (Ubuntu 22.04 LTS / 24.04 LTS recommended)
- **ROCm Stack**: ROCm 6.0, 6.1, or 6.2 installed in `/opt/rocm`
- **PyTorch with ROCm/HIP**:
  ```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.2
  ```
- **Device Permissions**:
  The user running ROCmHub must have access to the Kernel Fusion Driver (`/dev/kfd`) and DRI devices (`/dev/dri`):
  ```bash
  sudo usermod -aG render,video $USER
  newgrp render
  ```

---

## 3. The 7-Stage Pipeline Walkthrough

ROCmHub follows a deterministic, 7-stage pipeline:

```
[ Stage 1: Doctor ] ──> [ Stage 2: Inspect ] ──> [ Stage 3: Check ]
         │
         ▼
[ Stage 4: Run / Forge ] ──> [ Stage 5: Validate ] ──> [ Stage 6: Benchmark ]
                                                              │
                                                              ▼
                                                   [ Stage 7: Artifact & Guard ]
```

### Stage 1: Doctor Diagnostics (`rocmhub doctor`)
Inspects host kernel, AMD GPU hardware, `/dev/kfd` permissions, `/opt/rocm` stack, and PyTorch HIP availability.

```bash
# Terminal human-readable report
rocmhub doctor

# Structured JSON report
rocmhub doctor --json
```

*Exit Codes*: `0` (READY), `2` (NO_ACCELERATOR), `1` (WARNING / Incomplete).

### Stage 2: Model Static Inspection (`rocmhub inspect`)
Inspects Hugging Face repository metadata, resolves immutable 40-character Git commit SHA, parses architecture and parameter count without downloading weight tensors.

```bash
rocmhub inspect Qwen/Qwen2.5-0.5B-Instruct
```

### Stage 3: Preflight Capability Check (`rocmhub check`)
Evaluates model requirements against host hardware. Halts before any heavy disk or GPU activity if the host is incompatible.

```bash
rocmhub check Qwen/Qwen2.5-0.5B-Instruct
```

### Stage 4: Baseline Run or Dry-Run (`rocmhub run --dry-run`)
Executes baseline inference or inspects execution plan in dry-run mode:

```bash
# Safe dry-run (verifies plan without downloading weights or executing inference)
rocmhub run Qwen/Qwen2.5-0.5B-Instruct --dry-run

# Live baseline inference (requires live AMD GPU)
rocmhub run Qwen/Qwen2.5-0.5B-Instruct --precision fp16 --device 0
```

### Stage 5: Correctness Validation (`rocmhub validate`)
Evaluates model outputs against strict technical invariants (absence of numeric overflow strings, non-empty generation, valid token counts).

```bash
rocmhub validate Qwen/Qwen2.5-0.5B-Instruct --precision fp16
```

### Stage 6: Performance Benchmark (`rocmhub benchmark`)
Measures Time-To-First-Token (TTFT), Inter-Token Latency (ITL percentiles: mean, p50, p90, p99), throughput (tokens/s), and peak VRAM.

```bash
rocmhub benchmark Qwen/Qwen2.5-0.5B-Instruct --precision fp16 --warmup-runs 2 --runs 5
```

### Stage 7: Verifiable Artifact & BenchmarkGuard (`rocmhub artifact` & `rocmhub guard`)
Packages execution evidence into a signed, immutable artifact bundle and runs `BenchmarkGuard` to audit cryptographic checksums and evidence reproducibility.

```bash
# Verify artifact integrity
rocmhub artifact verify artifacts/art-<hash>

# Audit reproducibility with BenchmarkGuard
rocmhub guard verify artifacts/art-<hash>
```

---

## 4. Automated Reproducible Pipeline Script

ROCmHub includes a unified, fail-closed pipeline script [`scripts/first_amd_run.py`](scripts/first_amd_run.py).

### Running in Safe / Dry-Run Mode (Default)
Safe to run on any machine (macOS, Linux, CI). Validates all 7 stages, produces diagnostic JSON, and never downloads weights:

```bash
python3 scripts/first_amd_run.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### Running Live on AMD ROCm Hardware
Requires confirmed AMD GPU and explicit user consent via `--live-amd-consent`:

```bash
python3 scripts/first_amd_run.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --precision fp16 \
  --device 0 \
  --live-amd-consent
```

Diagnostics and artifact bundles are saved to `diagnostics/first_amd_run.json` and `diagnostics/artifacts/`.

---

## 5. Troubleshooting Common AMD ROCm Issues

| Symptom | Probable Cause | Resolution |
| :--- | :--- | :--- |
| `Doctor: kernel_kfd WARN` | `/dev/kfd` lacks read/write permissions | Add user to render/video: `sudo usermod -aG render,video $USER` and reboot or re-login. |
| `Doctor: pytorch_hip INFO` | Standard PyTorch installed without HIP | Install ROCm-enabled PyTorch wheel from PyTorch official index. |
| `HIP error: invalid device` | RDNA3 GPU unrecognized by default ROCm | Set target override: `export HSA_OVERRIDE_GFX_VERSION=11.0.0`. |
| `Memory access fault (OOM)` | Model exceeds available GPU VRAM | Select a smaller model or switch precision to `fp16` or `bf16`. |
| `Model weights not downloaded` | Host is in non-AMD diagnostic mode | This is intentional safe behavior. Connect to an AMD ROCm host to run live inference. |
