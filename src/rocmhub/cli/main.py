"""Main entrypoint for the rocmhub CLI."""

import argparse
import sys
from typing import List, Optional

from rocmhub import __version__


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
        nargs="?",
        default=None,
        help="Model repository or ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct').",
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
        sys.stderr.write(
            "rocmhub inspect: NOT_IMPLEMENTED (Scheduled for Phase 4: Model Source & Inspection)\n"
        )
        return 1

    if parsed_args.command == "run":
        sys.stderr.write(
            "rocmhub run: NOT_IMPLEMENTED (Scheduled for Phase 5-8: Runner, Benchmark, and Artifact Pipeline)\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
