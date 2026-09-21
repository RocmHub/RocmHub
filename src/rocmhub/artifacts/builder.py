"""ArtifactBuilder orchestrating end-to-end atomic bundle creation, secret scanning, and manifest generation."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rocmhub.artifacts.integrity import (
    canonical_json_bytes,
    compute_file_sha256,
    scan_for_secrets,
)
from rocmhub.artifacts.manifest import (
    build_manifest,
    build_reproduction_metadata,
    compute_artifact_id,
    compute_experiment_id,
)
from rocmhub.artifacts.storage import LocalArtifactStore
from rocmhub.core.types import (
    ArtifactFileEntry,
    ArtifactManifest,
    ArtifactStatus,
    BenchmarkResult,
    CapabilityReport,
    DetectionReport,
    ModelSpec,
    RunResult,
    ValidationReport,
)


class ArtifactBuilder:
    """Coordinates the compilation of multi-stage execution evidence into an immutable, verifiable artifact bundle."""

    def __init__(self, store: Optional[LocalArtifactStore] = None) -> None:
        self.store = store or LocalArtifactStore()

    @staticmethod
    def _write_canonical_file(target_file: Path, data: Any) -> None:
        """Serialize data to canonical JSON bytes and write to disk with flush."""
        bytes_content = canonical_json_bytes(data)
        with target_file.open("wb") as f:
            f.write(bytes_content)
            f.flush()
            os.fsync(f.fileno())

    def build(
        self,
        model: ModelSpec,
        detection: DetectionReport,
        capability: CapabilityReport,
        run: RunResult,
        benchmark: BenchmarkResult,
        validation: ValidationReport,
        raw_benchmark_data: Optional[Dict[str, Any] | List[Any]] = None,
        device_id: int = 0,
        precision: str = "fp16",
        runtime_name: str = "pytorch_transformers_hip",
        benchmark_params: Optional[Dict[str, Any]] = None,
        validation_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ArtifactManifest, Path]:
        """Compile execution evidence from all stages into an immutable artifact bundle.

        Lifecycle:
        1. Scan all input data for secrets (fails closed if secrets found).
        2. Stage bundle in an isolated temporary directory.
        3. Write canonical JSON for all component payloads.
        4. Calculate per-file cryptographic SHA-256 and byte sizes.
        5. Derive deterministic experiment_id and content-derived artifact_id.
        6. Assemble and sign manifest.json.
        7. Assemble detached checksums.json.
        8. Atomically commit staged directory into store.

        Args:
            model: Inspected ModelSpec.
            detection: Hardware and environment DetectionReport.
            capability: Preflight CapabilityReport.
            run: Baseline RunResult.
            benchmark: BenchmarkResult.
            validation: ValidationReport.
            raw_benchmark_data: Optional raw timing and iteration measurements.
            device_id: Accelerator device index.
            precision: Inference precision string.
            runtime_name: Name of the runtime adapter.
            benchmark_params: Parameters for benchmark workload.
            validation_params: Parameters for validation suite.

        Returns:
            Tuple of (signed ArtifactManifest, final artifact directory Path).

        Raises:
            SecretDetectedError: If credentials or secrets are detected in payload.
            ArtifactConflictError: If artifact exists with differing data.
            ArtifactPackagingError: If packaging or filesystem commit fails.
        """
        b_params = benchmark_params or {
            "prompt": run.prompt,
            "max_new_tokens": run.generation_params.get("max_new_tokens", 16),
            "warmup_runs": benchmark.warmup_runs,
            "measurement_runs": benchmark.measurement_runs_requested,
        }
        v_params = validation_params or {
            "mode": validation.mode.value,
            "cases_total": validation.cases_total,
        }

        # 1. Fail-closed secret scan across all input components
        scan_for_secrets(model.model_dump(mode="json"), path="model")
        scan_for_secrets(detection.model_dump(mode="json"), path="detection")
        scan_for_secrets(capability.model_dump(mode="json"), path="capability")
        scan_for_secrets(run.model_dump(mode="json"), path="run")
        scan_for_secrets(benchmark.model_dump(mode="json"), path="benchmark")
        scan_for_secrets(validation.model_dump(mode="json"), path="validation")
        if raw_benchmark_data is not None:
            scan_for_secrets(raw_benchmark_data, path="raw_benchmark")

        # 2. Derive canonical hardware summary and deterministic experiment_id
        gpus = detection.gpus
        hardware_summary = {
            "gpu_count": len(gpus),
            "devices": [
                {
                    "device_id": g.device_id,
                    "vendor": g.gpu_vendor,
                    "device_name": g.device_name,
                    "gfx_target": g.gfx_target,
                    "vram_total_mb": g.vram_total_mb,
                }
                for g in gpus
            ],
        }

        experiment_id = compute_experiment_id(
            model=model,
            hardware_summary=hardware_summary,
            runtime_name=runtime_name,
            precision=precision,
            benchmark_params=b_params,
            validation_params=v_params,
        )

        reproduction = build_reproduction_metadata(
            model=model,
            runtime_name=runtime_name,
            precision=precision,
            device_id=device_id,
            detection=detection,
            benchmark_params=b_params,
            validation_params=v_params,
        )

        # 3. Create isolated staging directory
        staging_dir = self.store.root_dir / f".staging_{os.getpid()}_{time.time_ns()}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 4. Write all payload component files in canonical JSON format
            payload_files: Dict[str, Any] = {
                "model.json": model,
                "environment.json": detection.environment,
                "capabilities.json": capability,
                "run.json": run,
                "benchmark.json": benchmark,
                "validation.json": validation,
                "reproduce.json": reproduction,
            }

            if raw_benchmark_data is not None:
                payload_files["benchmark_raw.json"] = raw_benchmark_data

            for filename, data in payload_files.items():
                self._write_canonical_file(staging_dir / filename, data)

            # 5. Compute cryptographic checksums and sizes for all payload files
            file_entries: Dict[str, ArtifactFileEntry] = {}
            file_checksums: Dict[str, str] = {}

            for filename in sorted(payload_files.keys()):
                file_path = staging_dir / filename
                sha = compute_file_sha256(file_path)
                size = file_path.stat().st_size
                file_entries[filename] = ArtifactFileEntry(
                    path=filename,
                    sha256=sha,
                    size_bytes=size,
                )
                file_checksums[filename] = sha

            # 6. Compute deterministic content-derived artifact_id
            artifact_id = compute_artifact_id(
                experiment_id=experiment_id,
                file_checksums=file_checksums,
            )

            # 7. Assemble and write manifest.json
            manifest = build_manifest(
                artifact_id=artifact_id,
                experiment_id=experiment_id,
                model=model,
                detection=detection,
                capability=capability,
                run=run,
                benchmark=benchmark,
                validation=validation,
                reproduction=reproduction,
                files=file_entries,
                runtime=runtime_name,
                precision=precision,
                status=ArtifactStatus.COMPLETE,
            )
            manifest_path = staging_dir / "manifest.json"
            self._write_canonical_file(manifest_path, manifest)

            # 8. Compute manifest.json SHA-256 and write detached checksums.json
            manifest_sha = compute_file_sha256(manifest_path)
            all_checksums = dict(file_checksums)
            all_checksums["manifest.json"] = manifest_sha

            checksums_payload = {
                "schema_version": "1.0.0",
                "artifact_id": artifact_id,
                "experiment_id": experiment_id,
                "files": dict(sorted(all_checksums.items())),
            }
            self._write_canonical_file(staging_dir / "checksums.json", checksums_payload)

            # 9. Atomically commit the staged bundle into the store
            final_path = self.store.commit_staged_bundle(
                staged_dir=staging_dir,
                artifact_id=artifact_id,
            )
            return manifest, final_path

        except Exception:
            # Clean up temporary staging directory on failure
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
