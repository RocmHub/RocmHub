"""Main entrypoint for the rocmhub CLI."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from rocmhub import __version__
from rocmhub.core.errors import ROCmHubError
from rocmhub.core.types import ModelSpec
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector


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
        help="Output detected environment as JSON.",
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

    # Command: run
    run_parser = subparsers.add_parser(
        "run",
        help="Execute model baseline and run benchmark pipeline.",
    )
    run_parser.add_argument(
        "--model",
        required=False,
        help="Model ID to execute.",
    )
    run_parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model inference precision.",
    )
    run_parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to save generated artifact manifest.",
    )

    return parser


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


def main(args: Optional[List[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help(sys.stderr)
        return 0

    if parsed_args.command == "env":
        sys.stderr.write(
            "rocmhub env: NOT_IMPLEMENTED (Scheduled for Phase 3: Hardware & Environment Detection)\n"
        )
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

    if parsed_args.command == "run":
        sys.stderr.write(
            "rocmhub run: NOT_IMPLEMENTED (Scheduled for Phase 5-8: Runner, Benchmark, and Artifact Pipeline)\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
