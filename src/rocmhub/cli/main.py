"""Main entrypoint for the rocmhub CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional

from rocmhub import __version__
from rocmhub.artifacts import ArtifactBuilder, LocalArtifactStore
from rocmhub.benchmarks import BenchmarkConfig, BenchmarkHarness
from rocmhub.capabilities import CapabilityEvaluator
from rocmhub.core.errors import ROCmHubError
from rocmhub.core.types import (
    ArtifactManifest,
    ArtifactVerificationResult,
    BenchmarkResult,
    CapabilityReport,
    DetectionReport,
    EvaluationVerdict,
    ExecutionStatus,
    GuardVerdict,
    ModelSpec,
    ReproducibilityReport,
    RunResult,
    ValidationMode,
    ValidationReport,
    ValidationVerdict,
)
from rocmhub.engineer import (
    AIEngineer,
    EngineerBudget,
    EngineerObjective,
    EngineerRequest,
    EngineerStatus,
    format_engineer_report_table,
)
from rocmhub.forge import (
    BuildManifest,
    ForgeExecutor,
    ForgePlan,
    ForgePlanner,
)
from rocmhub.guard import BenchmarkGuard
from rocmhub.hardware.detector import SystemObserver
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector
from rocmhub.optimization import (
    OptimizationExecutor,
    OptimizationRequest,
    format_optimization_report_table,
)
from rocmhub.runners import HuggingFaceRunner
from rocmhub.validation import DEFAULT_VALIDATION_CASES, ValidationConfig, ValidationEvaluator


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for rocmhub."""
    parser = argparse.ArgumentParser(
        prog="rocmhub",
        description="ROCmHub: Automated model preparation, optimization, and benchmarking for AMD GPUs and ROCm.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"rocmhub {__version__}",
        help="Show ROCmHub version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: env
    env_parser = subparsers.add_parser(
        "env",
        help="Detect and inspect AMD GPU hardware and ROCm software environment.",
    )
    env_parser.add_argument(
        "--json",
        action="store_true",
        help="Output detected environment as pure JSON.",
    )

    # Command: inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect model architecture, parameter count, and tensor requirements.",
    )
    inspect_parser.add_argument(
        "model_id",
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    inspect_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Output ModelSpec as pure JSON on stdout.",
    )

    # Command: check
    check_parser = subparsers.add_parser(
        "check",
        help="Evaluate baseline execution viability without downloading full weights or running inference.",
    )
    check_parser.add_argument(
        "model_id",
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    check_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output CapabilityReport as pure JSON on stdout.",
    )

    # Command: run
    run_parser = subparsers.add_parser(
        "run",
        help="Execute model baseline inference on target AMD GPU.",
    )
    run_parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    run_parser.add_argument(
        "--model",
        dest="model_opt",
        required=False,
        help="Alternative flag for model ID.",
    )
    run_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    run_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Target accelerator device index (default: 0).",
    )
    run_parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Inference precision (default: 'fp16').",
    )
    run_parser.add_argument(
        "--prompt",
        default="Hello, what are you?",
        help="Input text prompt for inference.",
    )
    run_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate (default: 16).",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Output RunResult as pure JSON on stdout.",
    )

    # Command: benchmark
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark model baseline performance (TTFT, ITL, throughput, VRAM).",
    )
    benchmark_parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    benchmark_parser.add_argument(
        "--model",
        dest="model_opt",
        required=False,
        help="Alternative flag for model ID.",
    )
    benchmark_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    benchmark_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Target accelerator device index (default: 0).",
    )
    benchmark_parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Inference precision (default: 'fp16').",
    )
    benchmark_parser.add_argument(
        "--prompt",
        default="Hello, ROCm!",
        help="Input text prompt for benchmark workload.",
    )
    benchmark_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate per run (default: 16).",
    )
    benchmark_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Count of untimed warmup iterations (default: 2).",
    )
    benchmark_parser.add_argument(
        "--runs",
        dest="runs",
        type=int,
        default=5,
        help="Count of timed measurement iterations (default: 5).",
    )
    benchmark_parser.add_argument(
        "--measurement-runs",
        dest="runs",
        type=int,
        help="Alias for --runs.",
    )
    benchmark_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BenchmarkResult as pure JSON on stdout.",
    )

    # Command: validate
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate model correctness: run the built-in validation suite (self-validation mode).",
    )
    validate_parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    validate_parser.add_argument(
        "--model",
        dest="model_opt",
        required=False,
        help="Alternative flag for model ID.",
    )
    validate_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    validate_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Target accelerator device index (default: 0).",
    )
    validate_parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Inference precision (default: 'fp16').",
    )
    validate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate per validation case (default: 16).",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output ValidationReport as pure JSON on stdout.",
    )

    # Command: artifact
    artifact_parser = subparsers.add_parser(
        "artifact",
        help="Manage, build, and verify reproducible artifact bundles.",
    )
    artifact_subparsers = artifact_parser.add_subparsers(
        dest="artifact_action",
        help="Artifact action to perform",
    )

    # Action: artifact build
    build_artifact_parser = artifact_subparsers.add_parser(
        "build",
        help="Run pipeline stages and compile an immutable reproducible artifact bundle.",
    )
    build_artifact_parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    build_artifact_parser.add_argument(
        "--model",
        dest="model_opt",
        required=False,
        help="Alternative flag for model ID.",
    )
    build_artifact_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    build_artifact_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Target accelerator device index (default: 0).",
    )
    build_artifact_parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Inference precision (default: 'fp16').",
    )
    build_artifact_parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to store compiled artifact bundles (default: 'artifacts').",
    )
    build_artifact_parser.add_argument(
        "--json",
        action="store_true",
        help="Output ArtifactManifest as pure JSON on stdout.",
    )

    # Action: artifact verify
    verify_artifact_parser = artifact_subparsers.add_parser(
        "verify",
        help="Verify cryptographic integrity, file inventory, and checksums of an artifact bundle.",
    )
    verify_artifact_parser.add_argument(
        "path",
        help="Filesystem path to the artifact bundle directory.",
    )
    verify_artifact_parser.add_argument(
        "--json",
        action="store_true",
        help="Output ArtifactVerificationResult as pure JSON on stdout.",
    )

    # Command: guard
    guard_parser = subparsers.add_parser(
        "guard",
        help="Evaluate benchmark reproducibility, sample variability, and artifact evidence truthfulness.",
    )
    guard_parser.add_argument(
        "path",
        help="Filesystem path to the artifact bundle directory or artifact ID.",
    )
    guard_parser.add_argument(
        "--json",
        action="store_true",
        help="Output ReproducibilityReport as pure JSON on stdout.",
    )

    # Command: forge
    forge_parser = subparsers.add_parser(
        "forge",
        help="Automated preparation and compilation of models for AMD GPUs.",
    )
    forge_subparsers = forge_parser.add_subparsers(
        dest="forge_action",
        help="Forge action to perform",
    )

    # Action: forge plan
    forge_plan_parser = forge_subparsers.add_parser(
        "plan",
        help="Generate a reproducible build plan for a model without downloading full weights.",
    )
    forge_plan_parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    forge_plan_parser.add_argument(
        "--model",
        dest="model_opt",
        required=False,
        help="Alternative flag for model ID.",
    )
    forge_plan_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    forge_plan_parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Target precision (default: 'fp16').",
    )
    forge_plan_parser.add_argument(
        "--target-gpu",
        default=None,
        help="Target AMD GPU architecture (e.g. gfx90a, gfx1100).",
    )
    forge_plan_parser.add_argument(
        "--output-dir",
        default=None,
        help="Target build output directory.",
    )
    forge_plan_parser.add_argument(
        "--recipe",
        default=None,
        help="Specific build recipe ID to use.",
    )
    forge_plan_parser.add_argument(
        "--json",
        action="store_true",
        help="Output ForgePlan as pure JSON on stdout.",
    )

    # Action: forge build
    forge_build_parser = forge_subparsers.add_parser(
        "build",
        help="Materialize model and generate verified build configuration for AMD ROCm.",
    )
    forge_build_parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    forge_build_parser.add_argument(
        "--model",
        dest="model_opt",
        required=False,
        help="Alternative flag for model ID.",
    )
    forge_build_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    forge_build_parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Target precision (default: 'fp16').",
    )
    forge_build_parser.add_argument(
        "--target-gpu",
        default=None,
        help="Target AMD GPU architecture (e.g. gfx90a, gfx1100).",
    )
    forge_build_parser.add_argument(
        "--output-dir",
        default=None,
        help="Target build output directory.",
    )
    forge_build_parser.add_argument(
        "--recipe",
        default=None,
        help="Specific build recipe ID to use.",
    )
    forge_build_parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing build output directory.",
    )
    forge_build_parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Materialize configs and metadata only, skipping tensor weights.",
    )
    forge_build_parser.add_argument(
        "--execute",
        action="store_true",
        help="Attempt real baseline execution on AMD GPU if available.",
    )
    forge_build_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BuildManifest as pure JSON on stdout.",
    )

    # Command: engineer
    engineer_parser = subparsers.add_parser(
        "engineer",
        help="Run autonomous AI Engineer to prepare, optimize, and evaluate models for AMD ROCm.",
    )
    engineer_parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Hugging Face model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    engineer_parser.add_argument(
        "--model",
        dest="model_opt",
        required=False,
        help="Alternative flag for model ID.",
    )
    engineer_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    engineer_parser.add_argument(
        "--target-gpu",
        default=None,
        help="Target AMD GPU architecture (e.g. gfx90a, gfx1100).",
    )
    engineer_parser.add_argument(
        "--objective",
        default="BASE_PREPARATION",
        choices=["BASE_PREPARATION", "FULL_PREPARATION", "AMD_EXECUTION", "MAX_THROUGHPUT", "MIN_LATENCY"],
        help="Engineering objective (default: 'BASE_PREPARATION').",
    )
    engineer_parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum number of loop attempts (default: 5).",
    )
    engineer_parser.add_argument(
        "--max-minutes",
        type=float,
        default=10.0,
        help="Maximum session execution time in minutes (default: 10.0).",
    )
    engineer_parser.add_argument(
        "--max-disk-gb",
        type=float,
        default=20.0,
        help="Maximum disk space in GB allowed (default: 20.0).",
    )
    engineer_parser.add_argument(
        "--allow-full-weights",
        action="store_true",
        help="Explicitly permit downloading full model weight tensors (default: False).",
    )
    engineer_parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom build output directory.",
    )
    engineer_parser.add_argument(
        "--json",
        action="store_true",
        help="Output EngineerReport as pure JSON on stdout.",
    )

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Generate, build, benchmark, and compare optimization candidate strategies on AMD ROCm.",
    )
    optimize_parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Hugging Face model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
    )
    optimize_parser.add_argument(
        "--model",
        dest="model_opt",
        required=False,
        help="Alternative flag for model ID.",
    )
    optimize_parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit revision (default: 'main').",
    )
    optimize_parser.add_argument(
        "--target-gpu",
        default=None,
        help="Target AMD GPU architecture (e.g. gfx90a, gfx1100).",
    )
    optimize_parser.add_argument(
        "--objective",
        default="MAX_THROUGHPUT",
        choices=["MAX_THROUGHPUT", "MIN_LATENCY", "BALANCED"],
        help="Optimization objective (default: 'MAX_THROUGHPUT').",
    )
    optimize_parser.add_argument(
        "--max-candidates",
        type=int,
        default=3,
        help="Maximum candidate strategies to plan and evaluate (default: 3).",
    )
    optimize_parser.add_argument(
        "--max-minutes",
        type=float,
        default=10.0,
        help="Maximum optimization session runtime in minutes (default: 10.0).",
    )
    optimize_parser.add_argument(
        "--allow-full-weights",
        action="store_true",
        help="Explicitly permit downloading full model weight tensors (default: False).",
    )
    optimize_parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory for candidate builds and report.",
    )
    optimize_parser.add_argument(
        "--json",
        action="store_true",
        help="Output OptimizationReport as pure JSON on stdout.",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start local ROCmHub backend HTTP server (FastAPI & Job Orchestrator).",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind server (default: '127.0.0.1').",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind server (default: 8000).",
    )
    serve_parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite database file.",
    )

    return parser


def _format_detection_report(report: DetectionReport) -> str:
    """Format DetectionReport into clean human-readable output."""
    env = report.environment
    lines = [
        f"{'System:':<16} {env.os}",
        f"{'Kernel:':<16} {env.kernel or 'None'}",
        f"{'Architecture:':<16} {env.architecture or 'None'}",
        f"{'Python:':<16} {env.python_version}",
        f"{'PyTorch:':<16} {env.torch_version}",
        f"{'ROCm:':<16} {env.rocm_version or 'not detected'}",
        f"{'HIP:':<16} {env.hip_version or 'not detected'}",
    ]

    if env.env_vars:
        lines.append("")
        lines.append("Active ROCm Environment Variables:")
        for k, v in sorted(env.env_vars.items()):
            lines.append(f"  {k} = {v}")

    lines.append("")
    gpu_count = len(report.gpus)
    lines.append(f"Detected GPUs: {gpu_count}")

    if gpu_count == 0:
        lines.append(f"{'Status:':<16} diagnostic / no compatible AMD runtime detected")
    else:
        for i, gpu in enumerate(report.gpus):
            lines.append("")
            lines.append(f"GPU {i}:")
            lines.append(f"  {'Vendor:':<16} {gpu.gpu_vendor or 'None'}")
            lines.append(f"  {'Device:':<16} {gpu.device_name or 'None'}")
            lines.append(f"  {'Device ID:':<16} {gpu.device_id if gpu.device_id is not None else 'None'}")
            if gpu.family:
                lines.append(f"  {'Family:':<16} {gpu.family}")
            lines.append(f"  {'gfx target:':<16} {gpu.gfx_target or 'None'}")
            vram_str = f"{gpu.vram_total_mb:,} MB" if gpu.vram_total_mb is not None else "None"
            lines.append(f"  {'VRAM:':<16} {vram_str}")
            cu_str = f"{gpu.compute_units}" if gpu.compute_units is not None else "None"
            lines.append(f"  {'Compute Units:':<16} {cu_str}")

    return "\n".join(lines)


def _format_model_spec(spec: ModelSpec) -> str:
    """Format ModelSpec into a readable summary table."""
    params_str = "None"
    if spec.parameter_count is not None:
        if spec.parameter_count >= 1_000_000_000:
            params_str = f"{spec.parameter_count:,} (~{spec.parameter_count / 1_000_000_000:.1f}B)"
        elif spec.parameter_count >= 1_000_000:
            params_str = f"{spec.parameter_count:,} (~{spec.parameter_count / 1_000_000:.1f}M)"
        else:
            params_str = f"{spec.parameter_count:,}"

    ctx_str = f"{spec.context_length:,}" if spec.context_length is not None else "None"
    dtype_str = spec.default_dtype or "None"
    arch_str = spec.architecture or "None"
    weights_str = spec.weights_format or "unknown"

    lines = [
        f"{'Model:':<22} {spec.model_id}",
        f"{'Requested revision:':<22} {spec.requested_revision}",
        f"{'Resolved commit SHA:':<22} {spec.commit_sha}",
        f"{'Architecture:':<22} {arch_str}",
        f"{'Parameter count:':<22} {params_str}",
        f"{'Context length:':<22} {ctx_str}",
        f"{'Default dtype:':<22} {dtype_str}",
        f"{'Weights format:':<22} {weights_str}",
    ]
    return "\n".join(lines)


def _format_capability_report(report: CapabilityReport) -> str:
    """Format CapabilityReport into a clean human-readable summary table."""
    lines: List[str] = []
    lines.append("Model:")
    lines.append(f"  {report.model.model_id}")
    lines.append(f"  Revision: {report.model.commit_sha}")
    if report.model.architecture:
        lines.append(f"  Architecture: {report.model.architecture}")
    lines.append("")

    lines.append("Environment:")
    lines.append(f"  ROCm:    {report.environment.rocm_version or 'not detected'}")
    lines.append(f"  HIP:     {report.environment.hip_version or 'not detected'}")
    lines.append(f"  PyTorch: {report.environment.torch_version}")
    lines.append("")

    if not report.hardware:
        lines.append("Detected GPUs: 0")
    else:
        for i, dev in enumerate(report.device_assessments):
            lines.append(f"GPU {dev.device_id if dev.device_id is not None else i}:")
            lines.append(f"  {dev.device_name or 'Unknown AMD GPU'}")
            lines.append(f"  gfx: {dev.gfx_target or 'None'}")
            if len(report.device_assessments) > 1:
                lines.append(f"  Verdict: {dev.verdict.value}")
            lines.append("")

    lines.append("Baseline candidate:")
    lines.append(f"  {report.capabilities.baseline_runtime_candidate or 'none'}")
    lines.append("")

    lines.append("Verdict:")
    lines.append(f"  {report.verdict.value}")
    lines.append("")

    lines.append("Reasons:")
    for r in report.reasons:
        sev_tag = f"[{r.severity.value.upper()}]"
        lines.append(f"  {sev_tag} {r.code}: {r.message}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def _format_run_result(result: RunResult) -> str:
    """Format RunResult into a clean human-readable summary table."""
    lines: List[str] = []
    lines.append("Model:")
    lines.append(f"  {result.model_id} (commit: {result.model_revision[:8]}...)")
    lines.append("")
    lines.append("Runtime:")
    dev_str = f"cuda:{result.device_id}" if result.device_id is not None else "None"
    lines.append(f"  {result.runtime_name} (device: {dev_str}, precision: {result.precision})")
    lines.append("")
    lines.append("Prompt:")
    lines.append(f"  {result.prompt}")
    lines.append("")
    lines.append("Generated Text:")
    lines.append(f"  {result.generated_text or '<none>'}")
    lines.append("")
    lines.append("Token Statistics:")
    lines.append(f"  Input tokens:     {result.input_tokens if result.input_tokens is not None else 'None'}")
    lines.append(f"  Generated tokens: {result.generated_tokens if result.generated_tokens is not None else 'None'}")
    lines.append("")
    lines.append(f"Status: {result.status.value}")
    if result.error:
        lines.append(f"Error:  {result.error}")
    return "\n".join(lines)


def _format_benchmark_result(res: BenchmarkResult) -> str:
    """Format BenchmarkResult into a clean human-readable summary table."""
    def _val(val: Optional[Any], unit: str = "") -> str:
        if val is None:
            return "not measured"
        if isinstance(val, float):
            return f"{val:,.2f} {unit}".strip()
        if isinstance(val, int):
            return f"{val:,} {unit}".strip()
        return str(val)

    lines = [
        f"{'Benchmark Result:':<24} {res.status.value}",
        f"{'Model:':<24} {res.model_id or 'unknown'}",
        f"{'Revision:':<24} {res.model_revision or 'unknown'}",
        f"{'Runtime:':<24} {res.runtime_name or 'pytorch_transformers_hip'}",
        f"{'Device:':<24} {f'cuda:{res.device_id}' if res.device_id is not None else 'cuda:0'}",
        f"{'Precision:':<24} {res.precision or 'fp16'}",
        "",
        f"{'Warmup Runs:':<24} {_val(res.warmup_runs)}",
        f"{'Measurement Runs:':<24} {_val(res.measurement_runs_completed)} of {_val(res.measurement_runs_requested)}",
        "",
        f"{'TTFT (median):':<24} {_val(res.ttft_ms, 'ms')}",
        f"{'ITL (mean):':<24} {_val(res.itl_ms_mean, 'ms')}",
        f"{'ITL (p50):':<24} {_val(res.itl_ms_p50, 'ms')}",
        f"{'ITL (p90):':<24} {_val(res.itl_ms_p90, 'ms')}",
        f"{'ITL (p99):':<24} {_val(res.itl_ms_p99, 'ms')}",
        f"{'Throughput:':<24} {_val(res.throughput_tokens_per_sec, 'tokens/sec')}",
        f"{'Total Latency:':<24} {_val(res.total_latency_ms, 'ms')}",
        f"{'Peak Allocator Memory:':<24} {_val(res.peak_vram_used_mb, 'MB')}",
    ]

    if res.error_message:
        lines.extend(["", f"{'Error Message:':<24} {res.error_message}"])

    return "\n".join(lines)


def _format_validation_report(report: ValidationReport) -> str:
    """Format ValidationReport into a clean human-readable summary table."""
    mode_str = report.mode.value
    verdict_str = report.verdict.value

    correctness_str: str
    if report.correctness_passed is None:
        correctness_str = "not measured"
    elif report.correctness_passed:
        correctness_str = "passed"
    else:
        correctness_str = "FAILED"

    qrr_str = f"{report.qrr_percent:.1f}%" if report.qrr_percent is not None else "n/a"

    lines = [
        f"{'Validation Result:':<26} {verdict_str}",
        f"{'Model:':<26} {report.model_id}",
        f"{'Baseline Revision:':<26} {report.baseline_revision}",
    ]
    if report.candidate_revision:
        lines.append(f"{'Candidate Revision:':<26} {report.candidate_revision}")

    lines += [
        f"{'Mode:':<26} {mode_str}",
        "",
        f"{'Cases Total:':<26} {report.cases_total}",
        f"{'Cases Completed:':<26} {report.cases_completed}",
        f"{'Cases Failed:':<26} {report.cases_failed}",
        f"{'Critical Cases Failed:':<26} {report.critical_cases_failed}",
        "",
        f"{'Correctness:':<26} {correctness_str}",
        f"{'Quality Measured:':<26} {'yes' if report.quality_measured else 'no'}",
        f"{'QRR:':<26} {qrr_str}",
        "",
        f"{'Verdict:':<26} {verdict_str}",
    ]

    if report.reasons:
        lines.append("")
        lines.append("Reasons:")
        for r in report.reasons:
            lines.append(f"  - {r}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")

    if report.case_results:
        lines.append("")
        lines.append("Case Results:")
        for cr in report.case_results:
            status_tag = cr.status.value.upper()
            tokens_str = (
                f"{cr.generated_tokens} tokens" if cr.generated_tokens is not None else "no tokens"
            )
            dur_str = f"{cr.run_duration_ms:.0f} ms" if cr.run_duration_ms is not None else "n/a"
            lines.append(f"  [{status_tag}] {cr.case_id} — {tokens_str} in {dur_str}")
            if cr.error:
                lines.append(f"           error: {cr.error}")

    return "\n".join(lines)


def _format_artifact_manifest(manifest: ArtifactManifest, path: Path | str) -> str:
    """Format ArtifactManifest into a clean human-readable summary table."""
    lines = [
        f"{'Artifact Bundle:':<24} {manifest.artifact_id}",
        f"{'Experiment ID:':<24} {manifest.experiment_id}",
        f"{'Status:':<24} {manifest.status.value}",
        f"{'Location:':<24} {path}",
        "",
        f"{'Model:':<24} {manifest.model.get('model_id', 'unknown')}",
        f"{'Revision:':<24} {manifest.model.get('immutable_revision', 'unknown')[:12]}...",
        f"{'Runtime:':<24} {manifest.runtime} ({manifest.precision})",
        "",
        f"{'Capability Verdict:':<24} {manifest.capability_verdict.value}",
        f"{'Execution Status:':<24} {manifest.execution_status.value}",
        f"{'Benchmark Status:':<24} {manifest.benchmark_status.value}",
        f"{'Validation Verdict:':<24} {manifest.validation_verdict.value}",
        "",
        "Files Inventory:",
    ]
    for rel_path, entry in sorted(manifest.files.items()):
        sha_short = entry.sha256[:12]
        lines.append(f"  {rel_path:<22} {entry.size_bytes:>8} B  (sha256: {sha_short}...)")

    return "\n".join(lines)


def _format_verification_result(res: ArtifactVerificationResult) -> str:
    """Format ArtifactVerificationResult into a clean human-readable summary table."""
    status_str = "VALID" if res.valid else "INVALID / TAMPERED"
    lines = [
        f"{'Verification:':<24} {status_str}",
        f"{'Artifact ID:':<24} {res.artifact_id}",
        f"{'Manifest Valid:':<24} {'yes' if res.manifest_valid else 'NO'}",
        f"{'Checksums Valid:':<24} {'yes' if res.checksums_valid else 'NO'}",
    ]

    if res.missing_files:
        lines.extend(["", "Missing Files:"])
        for f in res.missing_files:
            lines.append(f"  - {f}")

    if res.modified_files:
        lines.extend(["", "Modified / Corrupted Files:"])
        for f in res.modified_files:
            lines.append(f"  - {f}")

    if res.unexpected_files:
        lines.extend(["", "Unexpected Undeclared Files:"])
        for f in res.unexpected_files:
            lines.append(f"  - {f}")

    if res.errors:
        lines.extend(["", "Errors:"])
        for err in res.errors:
            lines.append(f"  - {err}")

    return "\n".join(lines)


def _format_reproducibility_report(report: ReproducibilityReport) -> str:
    """Format ReproducibilityReport into clean human-readable output."""
    verdict_str = report.verdict.value

    def _fmt_pct(val: Optional[float]) -> str:
        if val is None:
            return "n/a"
        return f"{val * 100.0:.2f}%"

    ref_str = "n/a" if report.reference_stable is None else ("stable" if report.reference_stable else "DRIFTED")

    lines = [
        f"{'Guard Verdict:':<26} {verdict_str}",
        f"{'Artifact ID:':<26} {report.artifact_id}",
        f"{'Experiment ID:':<26} {report.experiment_id}",
        f"{'Policy Version:':<26} {report.policy_version}",
        "",
        f"{'Environment Match:':<26} {'yes' if report.environment_consistent else 'DRIFT DETECTED'}",
        f"{'Hardware Consistent:':<26} {'yes' if report.hardware_consistent else 'THROTTLED / ERRORS'}",
        f"{'Benchmark Complete:':<26} {'yes' if report.benchmark_complete else 'no'}",
        f"{'Reference Calibration:':<26} {ref_str}",
        "",
        f"{'Measurement Runs:':<26} {report.valid_runs} valid of {report.measurement_runs} total",
        f"{'TTFT Variability (MAD):':<26} {_fmt_pct(report.ttft_variability)}",
        f"{'Throughput Var. (MAD):':<26} {_fmt_pct(report.throughput_variability)}",
        f"{'Latency Variability (MAD):':<26} {_fmt_pct(report.itl_variability)}",
        "",
        f"{'Final Verdict:':<26} {verdict_str}",
    ]

    if report.reasons:
        lines.append("")
        lines.append("Reasons:")
        for r in report.reasons:
            lines.append(f"  - {r}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def _format_forge_plan(plan: ForgePlan) -> str:
    """Format ForgePlan into a clean human-readable summary table."""
    lines = [
        "ROCmHub Model Forge Plan",
        "-" * 80,
        f"{'Plan ID:':<26} {plan.plan_id}",
        f"{'Model:':<26} {plan.model_id}",
        f"{'Revision:':<26} {plan.revision}",
        f"{'Target GPU:':<26} {plan.target_gpu or 'Auto-detected / None'}",
        f"{'Precision:':<26} {plan.precision}",
        f"{'Recipe:':<26} {plan.recipe_id} (v{plan.recipe_version})",
        f"{'Estimated Disk Space:':<26} {plan.estimated_disk_space_bytes / (1024**3):.2f} GB ({plan.estimated_disk_space_bytes:,} bytes)",
        f"{'Compatibility:':<26} {'CONFIRMED (READY)' if plan.compatibility_confirmed else 'NO AMD GPU / NOT CONFIRMED'}",
        f"{'Output Directory:':<26} {plan.output_dir}",
        "",
        "Planned Build Steps:",
    ]
    for i, step in enumerate(plan.steps, start=1):
        lines.append(f"  {i}. {step.name:<24} {step.description}")
    return "\n".join(lines)


def _format_build_manifest(manifest: BuildManifest) -> str:
    """Format BuildManifest into clean human-readable output."""
    lines = [
        "ROCmHub Model Forge Build",
        "-" * 80,
        f"{'Build ID:':<26} {manifest.build_id}",
        f"{'Plan ID:':<26} {manifest.plan_id}",
        f"{'Model:':<26} {manifest.model_id}",
        f"{'Revision:':<26} {manifest.revision}",
        f"{'Target GPU:':<26} {manifest.target_gpu or 'Auto-detected / None'}",
        f"{'Precision:':<26} {manifest.precision}",
        f"{'Recipe:':<26} {manifest.recipe_id} (v{manifest.recipe_version})",
        f"{'Runtime:':<26} {manifest.runtime}",
        f"{'Build Status:':<26} {manifest.status.value}",
        f"{'AMD Validated:':<26} {'yes' if manifest.amd_validated else 'no'}",
        f"{'Build Directory:':<26} {manifest.build_dir}",
        f"{'Weights Path:':<26} {manifest.weights_path or '(metadata only)'}",
        "",
        "Executed Steps:",
    ]
    for step in manifest.steps:
        status_tag = f"[{step.status.value}]"
        lines.append(f"  {status_tag:<11} {step.name:<24} ({step.duration_seconds:.3f}s) {step.message or ''}")

    if manifest.artifacts:
        lines.append("")
        lines.append("Generated Artifacts:")
        for fname, digest in sorted(manifest.artifacts.items()):
            digest_short = digest[:12] + "..." if len(digest) > 12 else digest
            lines.append(f"  - {fname:<24} (sha256: {digest_short})")

    return "\n".join(lines)


def main(args: Optional[List[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help(sys.stderr)
        return 0

    if parsed_args.command == "env":
        try:
            observer = SystemObserver()
            report = observer.observe()
            if parsed_args.json:
                sys.stdout.write(report.model_dump_json(indent=2) + "\n")
            else:
                sys.stdout.write(_format_detection_report(report) + "\n")
            return 0
        except Exception as exc:
            sys.stderr.write(f"Error during environment detection: {exc}\n")
            return 1

    if parsed_args.command == "inspect":
        try:
            source = HuggingFaceModelSource()
            inspector = ModelInspector(source=source)
            spec = inspector.inspect(
                model_id=parsed_args.model_id,
                revision=parsed_args.revision,
            )
            if parsed_args.json:
                sys.stdout.write(spec.model_dump_json(indent=2) + "\n")
            else:
                sys.stdout.write(_format_model_spec(spec) + "\n")
            return 0
        except ROCmHubError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected error: {exc}\n")
            return 1

    if parsed_args.command == "check":
        try:
            source = HuggingFaceModelSource()
            inspector = ModelInspector(source=source)
            model_spec = inspector.inspect(
                model_id=parsed_args.model_id,
                revision=parsed_args.revision,
            )

            observer = SystemObserver()
            detection = observer.observe()

            evaluator = CapabilityEvaluator()
            capability_report = evaluator.evaluate(model=model_spec, detection=detection)

            if parsed_args.json:
                sys.stdout.write(capability_report.model_dump_json(indent=2) + "\n")
            else:
                sys.stdout.write(_format_capability_report(capability_report) + "\n")

            verdict_exit_codes = {
                EvaluationVerdict.READY: 0,
                EvaluationVerdict.NO_ACCELERATOR: 2,
                EvaluationVerdict.BLOCKED: 3,
                EvaluationVerdict.UNKNOWN: 4,
            }
            return verdict_exit_codes.get(capability_report.verdict, 1)

        except ROCmHubError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected error: {exc}\n")
            return 1

    if parsed_args.command == "run":
        target_model_id = parsed_args.model_id or parsed_args.model_opt
        if not target_model_id:
            sys.stderr.write("Error: model_id must be provided to 'rocmhub run'.\n")
            return 1

        try:
            # 1. Inspect model metadata (resolves immutable commit SHA, no weights downloaded)
            source = HuggingFaceModelSource()
            inspector = ModelInspector(source=source)
            model_spec = inspector.inspect(
                model_id=target_model_id,
                revision=parsed_args.revision,
            )

            # 2. Observe system hardware and runtime environment
            observer = SystemObserver()
            detection = observer.observe()

            # 3. Preflight capability evaluation
            evaluator = CapabilityEvaluator()
            capability_report = evaluator.evaluate(model=model_spec, detection=detection)

            # 4. Check preflight verdict: ONLY proceed if READY!
            if capability_report.verdict != EvaluationVerdict.READY:
                error_msg = (
                    f"Baseline execution skipped: preflight verdict is {capability_report.verdict.value}."
                )
                skipped_result = RunResult(
                    status=ExecutionStatus.SKIPPED,
                    runtime_name="pytorch_transformers_hip",
                    model_id=model_spec.model_id,
                    model_revision=model_spec.commit_sha,
                    device_id=parsed_args.device,
                    precision=parsed_args.precision,
                    prompt=parsed_args.prompt,
                    generated_text=None,
                    input_tokens=None,
                    generated_tokens=None,
                    error=error_msg,
                    generation_params={"max_new_tokens": parsed_args.max_new_tokens, "do_sample": False},
                )

                if parsed_args.json:
                    sys.stdout.write(skipped_result.model_dump_json(indent=2) + "\n")
                else:
                    sys.stdout.write(
                        f"Preflight Check: {capability_report.verdict.value}\n"
                        f"{error_msg}\n\n"
                        "Reasons:\n"
                    )
                    for r in capability_report.reasons:
                        sys.stdout.write(f"  [{r.severity.value.upper()}] {r.code}: {r.message}\n")
                    sys.stdout.write("\nModel weights were NOT downloaded and inference was NOT executed.\n")

                verdict_exit_codes = {
                    EvaluationVerdict.NO_ACCELERATOR: 2,
                    EvaluationVerdict.BLOCKED: 3,
                    EvaluationVerdict.UNKNOWN: 4,
                }
                return verdict_exit_codes.get(capability_report.verdict, 1)

            # 5. Preflight is READY: proceed to execution with HuggingFaceRunner
            runner = HuggingFaceRunner()
            try:
                runner.load(
                    model=model_spec,
                    device_id=parsed_args.device,
                    precision=parsed_args.precision,
                )
                run_result = runner.generate(
                    prompt=parsed_args.prompt,
                    max_new_tokens=parsed_args.max_new_tokens,
                )
                if parsed_args.json:
                    sys.stdout.write(run_result.model_dump_json(indent=2) + "\n")
                else:
                    sys.stdout.write(_format_run_result(run_result) + "\n")
                return 0
            finally:
                runner.unload()

        except ROCmHubError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected error: {exc}\n")
            return 1

    if parsed_args.command == "benchmark":
        target_model_id = parsed_args.model_id or parsed_args.model_opt
        if not target_model_id:
            sys.stderr.write("Error: model_id must be provided to 'rocmhub benchmark'.\n")
            return 1

        try:
            # 1. Inspect model metadata (resolves immutable commit SHA, no weights downloaded)
            source = HuggingFaceModelSource()
            inspector = ModelInspector(source=source)
            model_spec = inspector.inspect(
                model_id=target_model_id,
                revision=parsed_args.revision,
            )

            # 2. Observe system hardware and runtime environment
            observer = SystemObserver()
            detection = observer.observe()

            # 3. Preflight capability evaluation
            evaluator = CapabilityEvaluator()
            capability_report = evaluator.evaluate(model=model_spec, detection=detection)

            # 4. Check preflight verdict: ONLY proceed if READY!
            if capability_report.verdict != EvaluationVerdict.READY:
                error_msg = (
                    f"Benchmark skipped: preflight verdict is {capability_report.verdict.value}."
                )
                skipped_bench_result = BenchmarkResult(
                    status=ExecutionStatus.SKIPPED,
                    error_message=error_msg,
                    model_id=model_spec.model_id,
                    model_revision=model_spec.commit_sha,
                    device_id=parsed_args.device,
                    runtime_name="pytorch_transformers_hip",
                    precision=parsed_args.precision,
                    warmup_runs=parsed_args.warmup_runs,
                    measurement_runs_requested=parsed_args.runs,
                    measurement_runs_completed=0,
                    failed_runs=0,
                )

                if parsed_args.json:
                    sys.stdout.write(skipped_bench_result.model_dump_json(indent=2) + "\n")
                else:
                    sys.stdout.write(
                        f"Preflight Check: {capability_report.verdict.value}\n"
                        f"{error_msg}\n\n"
                        "Reasons:\n"
                    )
                    for r in capability_report.reasons:
                        sys.stdout.write(f"  [{r.severity.value.upper()}] {r.code}: {r.message}\n")
                    sys.stdout.write("\nModel weights were NOT downloaded and benchmark was NOT executed.\n")

                verdict_exit_codes = {
                    EvaluationVerdict.NO_ACCELERATOR: 2,
                    EvaluationVerdict.BLOCKED: 3,
                    EvaluationVerdict.UNKNOWN: 4,
                }
                return verdict_exit_codes.get(capability_report.verdict, 1)

            # 5. Preflight is READY: proceed to benchmark execution
            bench_config = BenchmarkConfig(
                prompt=parsed_args.prompt,
                max_new_tokens=parsed_args.max_new_tokens,
                warmup_runs=parsed_args.warmup_runs,
                measurement_runs=parsed_args.runs,
                precision=parsed_args.precision,
                device_id=parsed_args.device,
            )

            runner = HuggingFaceRunner()
            try:
                runner.load(
                    model=model_spec,
                    device_id=parsed_args.device,
                    precision=parsed_args.precision,
                )
                harness = BenchmarkHarness(runner=runner, config=bench_config)
                benchmark_result = harness.run(model=model_spec)

                if parsed_args.json:
                    sys.stdout.write(benchmark_result.model_dump_json(indent=2) + "\n")
                else:
                    sys.stdout.write(_format_benchmark_result(benchmark_result) + "\n")

                return 0 if benchmark_result.status == ExecutionStatus.SUCCESS else 1
            finally:
                runner.unload()

        except ROCmHubError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected error: {exc}\n")
            return 1

    if parsed_args.command == "validate":
        target_model_id = parsed_args.model_id or parsed_args.model_opt
        if not target_model_id:
            sys.stderr.write("Error: model_id must be provided to 'rocmhub validate'.\n")
            return 1

        try:
            # 1. Inspect model metadata (resolves immutable commit SHA, no weights downloaded)
            source = HuggingFaceModelSource()
            inspector = ModelInspector(source=source)
            model_spec = inspector.inspect(
                model_id=target_model_id,
                revision=parsed_args.revision,
            )

            # 2. Observe system hardware and runtime environment
            observer = SystemObserver()
            detection = observer.observe()

            # 3. Preflight capability evaluation
            evaluator = CapabilityEvaluator()
            capability_report = evaluator.evaluate(model=model_spec, detection=detection)

            # 4. Preflight gate — skip execution if no capable accelerator available
            if capability_report.verdict != EvaluationVerdict.READY:
                skip_reason = (
                    f"Validation skipped: preflight verdict is {capability_report.verdict.value}."
                )
                # Build NOT_MEASURED report — no inference was executed
                not_measured_report = ValidationReport(
                    mode=ValidationMode.SELF_VALIDATION,
                    model_id=model_spec.model_id,
                    baseline_revision=model_spec.commit_sha,
                    candidate_revision=None,
                    verdict=ValidationVerdict.NOT_MEASURED,
                    correctness_passed=None,
                    quality_measured=False,
                    qrr_percent=None,
                    cases_total=len(DEFAULT_VALIDATION_CASES),
                    cases_completed=0,
                    cases_failed=0,
                    critical_cases_failed=0,
                    case_results=[],
                    reasons=[skip_reason]
                    + [f"[{r.severity.value.upper()}] {r.code}: {r.message}" for r in capability_report.reasons],
                    warnings=capability_report.warnings,
                )

                if parsed_args.json:
                    sys.stdout.write(not_measured_report.model_dump_json(indent=2) + "\n")
                else:
                    sys.stdout.write(
                        f"Preflight Check: {capability_report.verdict.value}\n"
                        f"{skip_reason}\n\n"
                        "Reasons:\n"
                    )
                    for r in capability_report.reasons:
                        sys.stdout.write(f"  [{r.severity.value.upper()}] {r.code}: {r.message}\n")
                    sys.stdout.write("\nModel weights were NOT downloaded and validation was NOT executed.\n")

                verdict_exit_codes = {
                    EvaluationVerdict.NO_ACCELERATOR: 2,
                    EvaluationVerdict.BLOCKED: 3,
                    EvaluationVerdict.UNKNOWN: 4,
                }
                return verdict_exit_codes.get(capability_report.verdict, 1)

            # 5. Preflight is READY: proceed to validation execution
            val_config = ValidationConfig(
                cases=DEFAULT_VALIDATION_CASES,
                max_new_tokens=parsed_args.max_new_tokens,
            )
            val_evaluator = ValidationEvaluator(config=val_config)

            runner = HuggingFaceRunner()
            try:
                runner.load(
                    model=model_spec,
                    device_id=parsed_args.device,
                    precision=parsed_args.precision,
                )
                validation_report = val_evaluator.run_self_validation(
                    runner=runner,
                    model=model_spec,
                )

                if parsed_args.json:
                    sys.stdout.write(validation_report.model_dump_json(indent=2) + "\n")
                else:
                    sys.stdout.write(_format_validation_report(validation_report) + "\n")

                # Exit codes: 0=PASS, 2=NOT_MEASURED, 3=FAIL, 4=INCONCLUSIVE
                verdict_exit_codes_val = {
                    ValidationVerdict.PASS: 0,
                    ValidationVerdict.NOT_MEASURED: 2,
                    ValidationVerdict.FAIL: 3,
                    ValidationVerdict.INCONCLUSIVE: 4,
                }
                return verdict_exit_codes_val.get(validation_report.verdict, 1)
            finally:
                runner.unload()

        except ROCmHubError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected error: {exc}\n")
            return 1

    if parsed_args.command == "artifact":
        if not getattr(parsed_args, "artifact_action", None):
            sys.stderr.write("Error: Subcommand required ('build' or 'verify'). Run 'rocmhub artifact --help'.\n")
            return 1

        if parsed_args.artifact_action == "verify":
            try:
                store = LocalArtifactStore()
                res = store.verify_artifact(parsed_args.path)
                if parsed_args.json:
                    sys.stdout.write(res.model_dump_json(indent=2) + "\n")
                else:
                    sys.stdout.write(_format_verification_result(res) + "\n")
                return 0 if res.valid else 1
            except Exception as exc:
                sys.stderr.write(f"Verification error: {exc}\n")
                return 1

        if parsed_args.artifact_action == "build":
            target_model_id = parsed_args.model_id or parsed_args.model_opt
            if not target_model_id:
                sys.stderr.write("Error: model_id must be provided to 'rocmhub artifact build'.\n")
                return 1

            try:
                # 1. Inspect model metadata (resolves immutable commit SHA without downloading weights)
                source = HuggingFaceModelSource()
                inspector = ModelInspector(source=source)
                model_spec = inspector.inspect(
                    model_id=target_model_id,
                    revision=parsed_args.revision,
                )

                # 2. Observe system hardware and runtime environment
                observer = SystemObserver()
                detection = observer.observe()

                # 3. Preflight capability evaluation
                evaluator = CapabilityEvaluator()
                capability_report = evaluator.evaluate(model=model_spec, detection=detection)

                store = LocalArtifactStore(root_dir=parsed_args.output_dir)
                builder = ArtifactBuilder(store=store)

                if capability_report.verdict != EvaluationVerdict.READY:
                    # Preflight check failed: construct honest diagnostic execution outcomes
                    # Zero weights are downloaded, zero GPU inference executed.
                    run_result = RunResult(
                        status=ExecutionStatus.SKIPPED,
                        runtime_name="pytorch_transformers_hip",
                        model_id=model_spec.model_id,
                        model_revision=model_spec.commit_sha,
                        device_id=parsed_args.device,
                        precision=parsed_args.precision,
                        prompt="Hello, ROCmHub diagnostic run",
                        generated_text=None,
                        input_tokens=None,
                        generated_tokens=None,
                        error=f"Execution skipped: preflight verdict is {capability_report.verdict.value}.",
                        generation_params={"max_new_tokens": 16, "do_sample": False},
                    )

                    benchmark_result = BenchmarkResult(
                        status=ExecutionStatus.SKIPPED,
                        error_message=f"Benchmark skipped: preflight verdict is {capability_report.verdict.value}.",
                        model_id=model_spec.model_id,
                        model_revision=model_spec.commit_sha,
                        device_id=parsed_args.device,
                        runtime_name="pytorch_transformers_hip",
                        precision=parsed_args.precision,
                        warmup_runs=2,
                        measurement_runs_requested=5,
                        measurement_runs_completed=0,
                        failed_runs=0,
                    )

                    validation_report = ValidationReport(
                        mode=ValidationMode.SELF_VALIDATION,
                        model_id=model_spec.model_id,
                        baseline_revision=model_spec.commit_sha,
                        candidate_revision=None,
                        verdict=ValidationVerdict.NOT_MEASURED,
                        correctness_passed=None,
                        quality_measured=False,
                        qrr_percent=None,
                        cases_total=len(DEFAULT_VALIDATION_CASES),
                        cases_completed=0,
                        cases_failed=0,
                        critical_cases_failed=0,
                        case_results=[],
                        reasons=[
                            f"Validation skipped: preflight verdict is {capability_report.verdict.value}."
                        ],
                    )

                    manifest, final_path = builder.build(
                        model=model_spec,
                        detection=detection,
                        capability=capability_report,
                        run=run_result,
                        benchmark=benchmark_result,
                        validation=validation_report,
                        device_id=parsed_args.device,
                        precision=parsed_args.precision,
                    )

                    if parsed_args.json:
                        sys.stdout.write(manifest.model_dump_json(indent=2) + "\n")
                    else:
                        sys.stdout.write(_format_artifact_manifest(manifest, final_path) + "\n")
                    return 0

                # Preflight check is READY: execute full inference pipeline
                runner = HuggingFaceRunner()
                try:
                    runner.load(
                        model=model_spec,
                        device_id=parsed_args.device,
                        precision=parsed_args.precision,
                    )
                    run_result = runner.generate(
                        prompt="Hello, ROCmHub execution!",
                        max_new_tokens=16,
                    )

                    bench_config = BenchmarkConfig(
                        prompt="Hello, ROCmHub benchmark!",
                        max_new_tokens=16,
                        warmup_runs=2,
                        measurement_runs=5,
                        precision=parsed_args.precision,
                        device_id=parsed_args.device,
                    )
                    harness = BenchmarkHarness(runner=runner, config=bench_config)
                    benchmark_result = harness.run(model=model_spec)

                    val_config = ValidationConfig(
                        cases=DEFAULT_VALIDATION_CASES,
                        max_new_tokens=16,
                    )
                    val_evaluator = ValidationEvaluator(config=val_config)
                    validation_report = val_evaluator.run_self_validation(
                        runner=runner,
                        model=model_spec,
                    )

                    manifest, final_path = builder.build(
                        model=model_spec,
                        detection=detection,
                        capability=capability_report,
                        run=run_result,
                        benchmark=benchmark_result,
                        validation=validation_report,
                        device_id=parsed_args.device,
                        precision=parsed_args.precision,
                    )

                    if parsed_args.json:
                        sys.stdout.write(manifest.model_dump_json(indent=2) + "\n")
                    else:
                        sys.stdout.write(_format_artifact_manifest(manifest, final_path) + "\n")
                    return 0
                finally:
                    runner.unload()

            except ROCmHubError as exc:
                sys.stderr.write(f"Error: {exc}\n")
                return 1
            except Exception as exc:
                sys.stderr.write(f"Unexpected error: {exc}\n")
                return 1

    if parsed_args.command == "guard":
        try:
            guard = BenchmarkGuard()
            guard_report = guard.evaluate(parsed_args.path)

            if parsed_args.json:
                sys.stdout.write(guard_report.model_dump_json(indent=2) + "\n")
            else:
                sys.stdout.write(_format_reproducibility_report(guard_report) + "\n")

            # Exit codes: 0=PASS, 2=NOT_MEASURED, 3=FAIL, 4=INCONCLUSIVE, 1=Error
            verdict_exit_codes_guard = {
                GuardVerdict.PASS: 0,
                GuardVerdict.NOT_MEASURED: 2,
                GuardVerdict.FAIL: 3,
                GuardVerdict.INCONCLUSIVE: 4,
            }
            return verdict_exit_codes_guard.get(guard_report.verdict, 1)
        except ROCmHubError as exc:
            sys.stderr.write(f"Guard error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected error in guard: {exc}\n")
            return 1

    if parsed_args.command == "forge":
        if not getattr(parsed_args, "forge_action", None):
            build_parser().parse_args(["forge", "--help"])
            return 0

        model_id = parsed_args.model_opt or parsed_args.model_id
        if not model_id:
            sys.stderr.write("Error: Model ID must be specified either as positional argument or via --model.\n")
            return 1

        if parsed_args.forge_action == "plan":
            try:
                planner = ForgePlanner()
                plan = planner.create_plan(
                    model_id=model_id,
                    revision=parsed_args.revision,
                    precision=parsed_args.precision,
                    target_gpu=parsed_args.target_gpu,
                    output_dir=parsed_args.output_dir,
                    recipe_id=parsed_args.recipe,
                )
                if parsed_args.json:
                    sys.stdout.write(plan.model_dump_json(indent=2) + "\n")
                else:
                    sys.stdout.write(_format_forge_plan(plan) + "\n")
                return 0
            except ROCmHubError as exc:
                sys.stderr.write(f"Forge plan error: {exc}\n")
                return 1
            except Exception as exc:
                sys.stderr.write(f"Unexpected error creating forge plan: {exc}\n")
                return 1

        if parsed_args.forge_action == "build":
            try:
                planner = ForgePlanner()
                plan = planner.create_plan(
                    model_id=model_id,
                    revision=parsed_args.revision,
                    precision=parsed_args.precision,
                    target_gpu=parsed_args.target_gpu,
                    output_dir=parsed_args.output_dir,
                    recipe_id=parsed_args.recipe,
                )
                executor = ForgeExecutor()
                build_manifest = executor.execute(
                    plan=plan,
                    download_weights=not parsed_args.no_weights,
                    force=parsed_args.force,
                    execute_inference=parsed_args.execute,
                )
                if parsed_args.json:
                    sys.stdout.write(build_manifest.to_json(indent=2) + "\n")
                else:
                    sys.stdout.write(_format_build_manifest(build_manifest) + "\n")
                return 0
            except ROCmHubError as exc:
                sys.stderr.write(f"Forge build error: {exc}\n")
                return 1
            except Exception as exc:
                sys.stderr.write(f"Unexpected error during forge build: {exc}\n")
                return 1

    if parsed_args.command == "engineer":
        model_id = parsed_args.model_opt or parsed_args.model_id
        if not model_id:
            sys.stderr.write("Error: Model ID must be specified either as positional argument or via --model.\n")
            return 1

        try:
            budget = EngineerBudget(
                max_attempts=parsed_args.max_attempts,
                max_execution_time_seconds=parsed_args.max_minutes * 60.0,
                max_disk_usage_bytes=int(parsed_args.max_disk_gb * (1024**3)),
                allow_full_weights=parsed_args.allow_full_weights,
            )
            request = EngineerRequest(
                model_id=model_id,
                revision=parsed_args.revision,
                target_gpu=parsed_args.target_gpu,
                objective=EngineerObjective(parsed_args.objective),
                budget=budget,
                output_dir=parsed_args.output_dir,
            )
            engineer = AIEngineer()
            engineer_report = engineer.run(request)

            if parsed_args.json:
                sys.stdout.write(engineer_report.model_dump_json(indent=2) + "\n")
            else:
                sys.stdout.write(format_engineer_report_table(engineer_report) + "\n")

            if engineer_report.status in (
                EngineerStatus.SUCCESS,
                EngineerStatus.STOPPED_ENVIRONMENT,
                EngineerStatus.CONFIG_ONLY,
            ):
                return 0
            elif engineer_report.status == EngineerStatus.BUDGET_EXCEEDED:
                return 2
            else:
                return 1
        except ROCmHubError as exc:
            sys.stderr.write(f"Engineer error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected error in AI Engineer: {exc}\n")
            return 1

    elif parsed_args.command == "optimize":
        model_id = parsed_args.model_id or parsed_args.model_opt
        if not model_id:
            sys.stderr.write("Error: Model ID must be specified either as positional argument or via --model.\n")
            return 1

        try:
            opt_request = OptimizationRequest(
                model_id=model_id,
                revision=parsed_args.revision,
                target_gpu=parsed_args.target_gpu,
                objective=parsed_args.objective,
                max_candidates=parsed_args.max_candidates,
                max_execution_time_seconds=parsed_args.max_minutes * 60.0,
                allow_full_weights=parsed_args.allow_full_weights,
                output_dir=parsed_args.output_dir,
            )
            opt_executor = OptimizationExecutor()
            opt_report = opt_executor.execute(opt_request)

            if parsed_args.json:
                sys.stdout.write(opt_report.model_dump_json(indent=2) + "\n")
            else:
                sys.stdout.write(format_optimization_report_table(opt_report) + "\n")

            if opt_report.status in ("SUCCESS", "CONFIG_ONLY", "STOPPED_ENVIRONMENT"):
                return 0
            elif opt_report.status == "BUDGET_EXCEEDED":
                return 2
            else:
                return 1
        except ROCmHubError as exc:
            sys.stderr.write(f"Optimization error: {exc}\n")
            return 1
        except Exception as exc:
            sys.stderr.write(f"Unexpected error in Optimization Engine: {exc}\n")
            return 1

    elif parsed_args.command == "serve":
        try:
            import uvicorn

            from rocmhub.server import ServerConfig, create_app

            config = ServerConfig(
                host=parsed_args.host,
                port=parsed_args.port,
            )
            if parsed_args.db_path:
                config.db_path = Path(parsed_args.db_path).resolve()

            app = create_app(config)
            uvicorn.run(app, host=parsed_args.host, port=parsed_args.port)
            return 0
        except Exception as exc:
            sys.stderr.write(f"Server error: {exc}\n")
            return 1

    return 0



if __name__ == "__main__":
    sys.exit(main())
