"""Main entrypoint for the rocmhub CLI."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from rocmhub import __version__
from rocmhub.capabilities import CapabilityEvaluator
from rocmhub.core.errors import ROCmHubError
from rocmhub.core.types import (
    CapabilityReport,
    DetectionReport,
    EvaluationVerdict,
    ExecutionStatus,
    ModelSpec,
    RunResult,
)
from rocmhub.hardware.detector import SystemObserver
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector
from rocmhub.runners import HuggingFaceRunner


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

    return 0


if __name__ == "__main__":
    sys.exit(main())
