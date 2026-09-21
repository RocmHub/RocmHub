"""Manifest assembly and deterministic identity generation for artifact bundles."""

from __future__ import annotations

from typing import Any, Dict

from rocmhub.artifacts.integrity import canonical_json_bytes, compute_sha256
from rocmhub.core.types import (
    ArtifactFileEntry,
    ArtifactManifest,
    ArtifactStatus,
    BenchmarkResult,
    CapabilityReport,
    DetectionReport,
    ModelSpec,
    ReproductionMetadata,
    RunResult,
    ValidationReport,
)


def compute_experiment_id(
    model: ModelSpec,
    hardware_summary: Dict[str, Any],
    runtime_name: str,
    precision: str,
    benchmark_params: Dict[str, Any],
    validation_params: Dict[str, Any],
) -> str:
    """Compute deterministic canonical identity for an experiment configuration.

    The experiment_id depends exclusively on input specifications:
    - Model identity and immutable commit SHA
    - Target hardware architecture and device configuration
    - Runtime adapter and precision
    - Benchmark configuration parameters
    - Validation suite configuration parameters

    Any change to precision, commit SHA, or benchmark parameters produces a different experiment_id.

    Args:
        model: Target ModelSpec.
        hardware_summary: Normalized hardware identity dictionary.
        runtime_name: Runtime adapter identifier.
        precision: Floating-point precision string.
        benchmark_params: Benchmark workload parameters.
        validation_params: Validation suite parameters.

    Returns:
        Formatted experiment ID: 'exp-<24-char-hex-hash>'
    """
    config_dict = {
        "model_id": model.model_id,
        "model_source": model.source,
        "commit_sha": model.commit_sha,
        "hardware": hardware_summary,
        "runtime_name": runtime_name,
        "precision": precision.lower().strip(),
        "benchmark_params": benchmark_params,
        "validation_params": validation_params,
    }
    digest = compute_sha256(canonical_json_bytes(config_dict))
    return f"exp-{digest[:24]}"


def compute_artifact_id(
    experiment_id: str,
    file_checksums: Dict[str, str],
) -> str:
    """Compute deterministic content-derived identifier for a materialized artifact bundle.

    The artifact_id is derived from the experiment_id and the cryptographic hashes
    of all materialized payload files in the bundle.

    Args:
        experiment_id: Deterministic experiment configuration ID.
        file_checksums: Mapping of relative file paths to their computed SHA-256 digests.

    Returns:
        Formatted artifact ID: 'art-<24-char-hex-hash>'
    """
    payload = {
        "experiment_id": experiment_id,
        "files": file_checksums,
    }
    digest = compute_sha256(canonical_json_bytes(payload))
    return f"art-{digest[:24]}"


def build_reproduction_metadata(
    model: ModelSpec,
    runtime_name: str,
    precision: str,
    device_id: int,
    detection: DetectionReport,
    benchmark_params: Dict[str, Any],
    validation_params: Dict[str, Any],
) -> ReproductionMetadata:
    """Build structured, declarative reproduction metadata for the experiment."""
    env = detection.environment
    gpus = detection.gpus

    gpu_req: Dict[str, Any] = {"device_id": device_id}
    if gpus and 0 <= device_id < len(gpus):
        target_gpu = gpus[device_id]
        gpu_req.update(
            {
                "vendor": target_gpu.gpu_vendor,
                "device_name": target_gpu.device_name,
                "gfx_target": target_gpu.gfx_target,
                "min_vram_mb": target_gpu.vram_total_mb,
            }
        )
    else:
        gpu_req.update({"vendor": "none", "device_name": "diagnostic_cpu"})

    return ReproductionMetadata(
        model_id=model.model_id,
        requested_revision=model.requested_revision,
        immutable_revision=model.commit_sha,
        runtime=runtime_name,
        precision=precision,
        device_requirements=gpu_req,
        benchmark_config=benchmark_params,
        validation_config=validation_params,
        environment_requirements={
            "os": env.os,
            "architecture": env.architecture,
            "python_version": env.python_version,
            "torch_version": env.torch_version,
            "rocm_version": env.rocm_version,
            "hip_version": env.hip_version,
        },
    )


def build_manifest(
    artifact_id: str,
    experiment_id: str,
    model: ModelSpec,
    detection: DetectionReport,
    capability: CapabilityReport,
    run: RunResult,
    benchmark: BenchmarkResult,
    validation: ValidationReport,
    reproduction: ReproductionMetadata,
    files: Dict[str, ArtifactFileEntry],
    runtime: str,
    precision: str,
    status: ArtifactStatus = ArtifactStatus.COMPLETE,
) -> ArtifactManifest:
    """Assemble the top-level ArtifactManifest recording all experiment evidence and file inventories."""
    env = detection.environment
    gpus = detection.gpus

    hardware_summary: Dict[str, Any] = {
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

    environment_summary: Dict[str, Any] = {
        "os": env.os,
        "kernel": env.kernel,
        "architecture": env.architecture,
        "python_version": env.python_version,
        "torch_version": env.torch_version,
        "rocm_version": env.rocm_version,
        "hip_version": env.hip_version,
    }

    model_summary: Dict[str, Any] = {
        "model_id": model.model_id,
        "source": model.source,
        "requested_revision": model.requested_revision,
        "immutable_revision": model.commit_sha,
        "architecture": model.architecture,
        "parameter_count": model.parameter_count,
        "context_length": model.context_length,
        "default_dtype": model.default_dtype,
        "weights_format": model.weights_format,
    }

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        experiment_id=experiment_id,
        status=status,
        model=model_summary,
        hardware=hardware_summary,
        environment=environment_summary,
        runtime=runtime,
        precision=precision,
        capability_verdict=capability.verdict,
        execution_status=run.status,
        benchmark_status=benchmark.status,
        validation_verdict=validation.verdict,
        files=files,
        reproduction=reproduction.model_dump(mode="json"),
    )

    manifest.sign_manifest()
    return manifest
