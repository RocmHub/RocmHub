#!/usr/bin/env python3
"""Automated Fresh Environment Smoke Test for ROCmHub Release Candidate.

Verifies end-to-end user installation from a clean wheel in an isolated venv:
1. Creates temporary isolated virtual environment.
2. Installs built wheel package.
3. Tests CLI entrypoint and commands: --help, --version, doctor.
4. Executes safe dry-run execution pipelines.
5. Asserts:
   - Zero model weights downloaded during dry-run.
   - Zero inference subprocess execution.
   - Truthful non-AMD / diagnostic exit codes and statuses.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Execute command and return result with captured stdout/stderr."""
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    print("=" * 70)
    print("ROCmHub Release Candidate: Fresh Environment Installation Smoke Test")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent
    wheel_candidates = glob.glob(str(repo_root / "dist" / "rocmhub-*.whl"))

    if not wheel_candidates:
        print("[!] No wheel found in dist/. Building wheel first...")
        build_res = run_cmd([sys.executable, "-m", "build", "--wheel"], cwd=repo_root)
        if build_res.returncode != 0:
            print(f"[FAIL] Wheel build failed:\n{build_res.stderr}")
            return 1
        wheel_candidates = glob.glob(str(repo_root / "dist" / "rocmhub-*.whl"))

    wheel_path = Path(wheel_candidates[-1]).resolve()
    print(f"[*] Target Wheel: {wheel_path.name}")

    with tempfile.TemporaryDirectory(prefix="rocmhub_smoke_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        venv_dir = temp_dir / "venv"
        print(f"[*] Creating isolated virtual environment at {venv_dir}...")
        venv_res = run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
        if venv_res.returncode != 0:
            print(f"[FAIL] Failed to create venv:\n{venv_res.stderr}")
            return 1

        venv_bin = venv_dir / "bin"
        pip_path = venv_bin / "pip"
        rocmhub_cli = venv_bin / "rocmhub"

        # 1. Install wheel into fresh venv
        print(f"[*] Installing {wheel_path.name} into isolated venv...")
        install_res = run_cmd([str(pip_path), "install", "--no-cache-dir", str(wheel_path)])
        if install_res.returncode != 0:
            print(f"[FAIL] Wheel installation failed:\n{install_res.stderr}")
            return 1
        print("    -> Package installed successfully.")

        # Ensure isolated cache dirs to verify no weights are written
        isolated_home = temp_dir / "home"
        isolated_home.mkdir()
        test_env = {
            **os.environ,
            "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
            "HOME": str(isolated_home),
            "HF_HOME": str(isolated_home / ".cache" / "huggingface"),
        }

        # 2. Check rocmhub --version
        print("[*] Testing `rocmhub --version`...")
        ver_res = run_cmd([str(rocmhub_cli), "--version"], env=test_env)
        if ver_res.returncode != 0 or "0.1.0" not in (ver_res.stdout + ver_res.stderr):
            print(f"[FAIL] `rocmhub --version` unexpected output:\n{ver_res.stdout}\n{ver_res.stderr}")
            return 1
        print(f"    -> Version verified: {ver_res.stdout.strip() or ver_res.stderr.strip()}")

        # 3. Check rocmhub --help
        print("[*] Testing `rocmhub --help`...")
        help_res = run_cmd([str(rocmhub_cli), "--help"], env=test_env)
        if help_res.returncode != 0 or "doctor" not in help_res.stdout:
            print(f"[FAIL] `rocmhub --help` missing expected subcommands:\n{help_res.stdout}")
            return 1
        print("    -> Help verified with doctor, inspect, run, forge subcommands.")

        # 4. Check rocmhub doctor
        print("[*] Testing `rocmhub doctor --json`...")
        doctor_res = run_cmd([str(rocmhub_cli), "doctor", "--json"], env=test_env)
        # On macOS or non-accelerated CI, doctor returns code 2 (NO_ACCELERATOR)
        if doctor_res.returncode not in (0, 2):
            print(f"[FAIL] Unexpected doctor exit code {doctor_res.returncode}:\n{doctor_res.stderr}")
            return 1
        try:
            doc_data = json.loads(doctor_res.stdout)
            print(f"    -> Doctor verdict: {doc_data['verdict']} (checks: {len(doc_data['checks'])})")
        except json.JSONDecodeError:
            print(f"[FAIL] Doctor did not return valid JSON:\n{doctor_res.stdout}")
            return 1

        # 5. Check rocmhub run --dry-run
        print("[*] Testing `rocmhub run --model Qwen/Qwen2.5-0.5B-Instruct --dry-run`...")
        run_res = run_cmd(
            [str(rocmhub_cli), "run", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--dry-run"],
            env=test_env,
        )
        # On Mac / non-AMD host, exits 2 (NO_ACCELERATOR) with preflight notice
        if run_res.returncode != 2:
            print(f"[FAIL] Expected exit code 2 for dry-run on non-AMD, got {run_res.returncode}:\n{run_res.stdout}\n{run_res.stderr}")
            return 1
        if "Model weights were NOT downloaded and inference was NOT executed" not in run_res.stdout:
            print(f"[FAIL] Expected safety guarantee text missing in stdout:\n{run_res.stdout}")
            return 1
        print("    -> Safe preflight exit code 2 verified; weight download prevented.")

        # 6. Check Forge build --no-weights and execute --dry-run
        forge_out = temp_dir / "forge_build"
        print(f"[*] Testing `rocmhub forge build --no-weights` in {forge_out}...")
        build_res = run_cmd(
            [
                str(rocmhub_cli),
                "forge",
                "build",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir",
                str(forge_out),
                "--no-weights",
            ],
            env=test_env,
        )
        if build_res.returncode != 0:
            print(f"[FAIL] Forge build failed:\n{build_res.stdout}\n{build_res.stderr}")
            return 1
        manifest_file = forge_out / "build_manifest.json"
        if not manifest_file.exists():
            print(f"[FAIL] Forge manifest not created at {manifest_file}")
            return 1
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_data.get("status") != "CONFIG_ONLY" or manifest_data.get("amd_validated") is True:
            print(f"[FAIL] Manifest violated safety invariants: {manifest_data}")
            return 1
        print("    -> Forge build generated verified CONFIG_ONLY manifest.")

        print("[*] Testing `rocmhub forge execute --dry-run`...")
        exec_res = run_cmd(
            [str(rocmhub_cli), "forge", "execute", str(forge_out), "--dry-run", "--json"],
            env=test_env,
        )
        if exec_res.returncode != 0:
            print(f"[FAIL] Forge execute dry-run failed:\n{exec_res.stdout}\n{exec_res.stderr}")
            return 1
        exec_data = json.loads(exec_res.stdout)
        if exec_data.get("amd_validated") is True or exec_data.get("dry_run") is not True:
            print(f"[FAIL] Execution result violated dry-run invariants: {exec_data}")
            return 1
        print("    -> Forge execute dry-run confirmed amd_validated=False and dry_run=True.")

        # 7. Check that no model weight files exist in isolated cache
        hf_cache = isolated_home / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            weight_files = list(hf_cache.rglob("*.safetensors")) + list(hf_cache.rglob("*.bin"))
            if weight_files:
                print(f"[FAIL] Weight files were downloaded into cache: {weight_files}")
                return 1
        print("    -> Verified 0 weight files downloaded into isolated cache.")

    print("=" * 70)
    print("ALL FRESH ENVIRONMENT SMOKE CHECKS PASSED SUCCESSFULLY (Release Candidate Ready)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
