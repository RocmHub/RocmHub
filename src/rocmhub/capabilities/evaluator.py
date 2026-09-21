"""CapabilityEvaluator: preflight assessment linking ModelSpec and DetectionReport."""

from __future__ import annotations

from typing import List, Optional

from rocmhub.capabilities.policy import CapabilityPolicy
from rocmhub.core.types import (
    CapabilityReport,
    DetectionReport,
    DeviceCapabilityAssessment,
    EvaluationReason,
    EvaluationSeverity,
    EvaluationVerdict,
    ModelSpec,
    SystemCapabilities,
)


class CapabilityEvaluator:
    """Evaluates whether system conditions and model characteristics warrant attempting baseline execution.

    Core Invariants:
    1. Preflight assessment only: answers "Are there grounds to attempt baseline execution?",
       NOT a proof that model execution will succeed or benchmark will pass.
    2. Zero weight downloads: uses static metadata only.
    3. Zero inference executions: does not construct tensors or call forward passes.
    4. Observation != Policy: separates factual discovery (SystemObserver) from
       interpretation (CapabilityPolicy).
    5. Non-binary verdicts: READY, BLOCKED, UNKNOWN, NO_ACCELERATOR.
    6. UNKNOWN never automatically becomes BLOCKED: new/unlisted GPUs remain UNKNOWN.
    7. Multi-GPU awareness: evaluates each device individually without merging into a virtual GPU.
    """

    def __init__(self, policy: Optional[CapabilityPolicy] = None) -> None:
        self._policy = policy or CapabilityPolicy()

    @property
    def policy(self) -> CapabilityPolicy:
        """Active capability evaluation policy."""
        return self._policy

    def evaluate(
        self,
        model: ModelSpec,
        detection: DetectionReport,
    ) -> CapabilityReport:
        """Perform preflight evaluation on model specification and environment detection.

        Args:
            model: Inspected ModelSpec.
            detection: Discovered DetectionReport from SystemObserver.

        Returns:
            CapabilityReport bundling structured reasons, conservative capabilities,
            per-device assessments, and overall verdict.
        """
        env = detection.environment
        gpus = detection.gpus

        # 1. Conservative factual evaluations
        amd_gpus = [g for g in gpus if (g.gpu_vendor == "AMD" or g.gpu_present)]
        amd_gpu_present = len(amd_gpus) > 0
        rocm_detected = env.rocm_version is not None
        hip_detected = env.hip_version is not None
        torch_available = env.torch_version != "not_installed"
        torch_hip_available = env.torch_hip_available
        remote_code_required = getattr(model, "remote_code_required", False)

        # Completeness of core model metadata
        missing_metadata: List[str] = []
        if model.architecture is None:
            missing_metadata.append("architecture")
        if model.parameter_count is None:
            missing_metadata.append("parameter_count")
        if model.context_length is None:
            missing_metadata.append("context_length")
        model_metadata_complete = len(missing_metadata) == 0

        # 2. Check for NO_ACCELERATOR (Diagnostic mode / macOS / CI / CPU-only host)
        if not amd_gpu_present:
            reasons: List[EvaluationReason] = [
                EvaluationReason(
                    code="NO_AMD_GPU",
                    severity=EvaluationSeverity.INFO,
                    message="No AMD GPU detected",
                    evidence={"detected_gpus": len(gpus)},
                )
            ]
            if not rocm_detected:
                reasons.append(
                    EvaluationReason(
                        code="ROCM_NOT_DETECTED",
                        severity=EvaluationSeverity.INFO,
                        message="ROCm runtime was not detected",
                        evidence={"rocm_version": None},
                    )
                )
            if not hip_detected:
                reasons.append(
                    EvaluationReason(
                        code="HIP_NOT_DETECTED",
                        severity=EvaluationSeverity.INFO,
                        message="HIP runtime was not detected",
                        evidence={"hip_version": None},
                    )
                )
            if not torch_hip_available:
                reasons.append(
                    EvaluationReason(
                        code="TORCH_HIP_UNAVAILABLE",
                        severity=EvaluationSeverity.INFO,
                        message="PyTorch HIP backend is not available",
                        evidence={"torch_version": env.torch_version, "torch_hip_available": False},
                    )
                )

            capabilities = SystemCapabilities(
                amd_gpu_present=False,
                rocm_detected=rocm_detected,
                hip_detected=hip_detected,
                torch_available=torch_available,
                torch_hip_available=torch_hip_available,
                model_metadata_complete=model_metadata_complete,
                remote_code_required=remote_code_required,
                baseline_runtime_candidate=None,
            )

            return CapabilityReport(
                model=model,
                environment=env,
                hardware=gpus,
                device_assessments=[],
                verdict=EvaluationVerdict.NO_ACCELERATOR,
                reasons=reasons,
                warnings=detection.warnings,
                capabilities=capabilities,
            )

        # 3. AMD Accelerator is present: evaluate environment blockers & model constraints
        reasons = []
        warnings = list(detection.warnings)
        env_blockers: List[EvaluationReason] = []

        reasons.append(
            EvaluationReason(
                code="AMD_GPU_PRESENT",
                severity=EvaluationSeverity.OK,
                message="AMD GPU hardware detected",
                evidence={"detected_gpus": len(amd_gpus)},
            )
        )

        # Check Model-level constraints
        if remote_code_required:
            r = EvaluationReason(
                code="REMOTE_CODE_REQUIRED",
                severity=EvaluationSeverity.BLOCKER,
                message="Model requires executing untrusted remote repository code",
                evidence={"remote_code_required": True},
            )
            reasons.append(r)
            env_blockers.append(r)

        if model.weights_format in self._policy.unsupported_weight_formats:
            r = EvaluationReason(
                code="UNSUPPORTED_WEIGHTS_FORMAT",
                severity=EvaluationSeverity.BLOCKER,
                message=f"Weights format '{model.weights_format}' is not supported by PyTorch baseline runner",
                evidence={"weights_format": model.weights_format},
            )
            reasons.append(r)
            env_blockers.append(r)

        if not model_metadata_complete:
            reasons.append(
                EvaluationReason(
                    code="MODEL_METADATA_INCOMPLETE",
                    severity=EvaluationSeverity.WARNING,
                    message=f"Optional model metadata fields are missing: {', '.join(missing_metadata)}",
                    evidence={"missing_fields": missing_metadata},
                )
            )
            warnings.append(f"Model metadata missing optional fields: {', '.join(missing_metadata)}")
        else:
            reasons.append(
                EvaluationReason(
                    code="MODEL_METADATA_VALID",
                    severity=EvaluationSeverity.OK,
                    message="Model metadata is complete",
                    evidence={"architecture": model.architecture, "parameter_count": model.parameter_count},
                )
            )

        # Check Environment-level constraints
        if rocm_detected:
            reasons.append(
                EvaluationReason(
                    code="ROCM_DETECTED",
                    severity=EvaluationSeverity.OK,
                    message="ROCm runtime detected",
                    evidence={"rocm_version": env.rocm_version},
                )
            )
        else:
            r = EvaluationReason(
                code="ROCM_NOT_DETECTED",
                severity=EvaluationSeverity.BLOCKER,
                message="ROCm runtime was not detected on system with AMD GPU",
                evidence={"rocm_version": None},
            )
            reasons.append(r)
            env_blockers.append(r)

        if hip_detected:
            reasons.append(
                EvaluationReason(
                    code="HIP_DETECTED",
                    severity=EvaluationSeverity.OK,
                    message="HIP runtime detected",
                    evidence={"hip_version": env.hip_version},
                )
            )
        elif rocm_detected:
            warnings.append("HIP runtime version could not be determined")

        if not torch_available:
            r = EvaluationReason(
                code="TORCH_NOT_INSTALLED",
                severity=EvaluationSeverity.BLOCKER,
                message="PyTorch is not installed in the current environment",
                evidence={"torch_version": "not_installed"},
            )
            reasons.append(r)
            env_blockers.append(r)
        elif not torch_hip_available:
            r = EvaluationReason(
                code="TORCH_NO_HIP",
                severity=EvaluationSeverity.BLOCKER,
                message="PyTorch is installed without functional ROCm/HIP backend support",
                evidence={"torch_version": env.torch_version, "torch_hip_available": False},
            )
            reasons.append(r)
            env_blockers.append(r)
        else:
            reasons.append(
                EvaluationReason(
                    code="TORCH_HIP_AVAILABLE",
                    severity=EvaluationSeverity.OK,
                    message="PyTorch ROCm/HIP backend is functional",
                    evidence={"torch_version": env.torch_version, "torch_hip_available": True},
                )
            )

        # 4. Multi-GPU Device-by-Device Assessment
        device_assessments: List[DeviceCapabilityAssessment] = []
        for idx, gpu in enumerate(amd_gpus):
            dev_reasons: List[EvaluationReason] = []
            dev_warnings: List[str] = []

            if env_blockers:
                # Inherit system/model blockers
                dev_verdict = EvaluationVerdict.BLOCKED
                dev_reasons.extend(env_blockers)
            else:
                # System environment is sound; inspect hardware target
                target = (gpu.gfx_target or "").lower().strip()
                if target and target in self._policy.known_gfx_targets:
                    dev_verdict = EvaluationVerdict.READY
                    dev_reasons.append(
                        EvaluationReason(
                            code="GFX_TARGET_SUPPORTED",
                            severity=EvaluationSeverity.OK,
                            message=f"AMD GFX target '{gpu.gfx_target}' is in known supported architecture set",
                            evidence={"gfx_target": gpu.gfx_target, "device_name": gpu.device_name},
                        )
                    )
                elif target and target not in ("unknown", "none", ""):
                    # New or unlisted AMD architecture: UNKNOWN, NOT BLOCKED!
                    dev_verdict = EvaluationVerdict.UNKNOWN
                    dev_reasons.append(
                        EvaluationReason(
                            code="UNKNOWN_GFX_TARGET",
                            severity=EvaluationSeverity.WARNING,
                            message=(
                                f"AMD GFX target '{gpu.gfx_target}' is not in policy reference set. "
                                "Compatibility cannot be confirmed statically, but baseline attempt is NOT blocked."
                            ),
                            evidence={"gfx_target": gpu.gfx_target, "device_name": gpu.device_name},
                        )
                    )
                    dev_warnings.append(f"GPU {idx} has unlisted GFX target: {gpu.gfx_target}")
                else:
                    # Target indeterminate
                    dev_verdict = EvaluationVerdict.UNKNOWN
                    dev_reasons.append(
                        EvaluationReason(
                            code="GFX_TARGET_INDETERMINATE",
                            severity=EvaluationSeverity.WARNING,
                            message="GPU GFX target architecture could not be determined. Insufficient data.",
                            evidence={"gfx_target": None, "device_name": gpu.device_name},
                        )
                    )
                    dev_warnings.append(f"GPU {idx} GFX target could not be determined")

            assessment = DeviceCapabilityAssessment(
                device_id=gpu.device_id if gpu.device_id is not None else idx,
                device_name=gpu.device_name,
                gfx_target=gpu.gfx_target,
                verdict=dev_verdict,
                reasons=dev_reasons,
                warnings=dev_warnings,
            )
            device_assessments.append(assessment)

        # 5. Determine Overall System Verdict
        if env_blockers:
            overall_verdict = EvaluationVerdict.BLOCKED
            baseline_candidate = None
        elif any(a.verdict == EvaluationVerdict.READY for a in device_assessments):
            overall_verdict = EvaluationVerdict.READY
            baseline_candidate = "pytorch_transformers_hip"
            reasons.append(
                EvaluationReason(
                    code="BASELINE_CANDIDATE_READY",
                    severity=EvaluationSeverity.OK,
                    message="Baseline runtime 'pytorch_transformers_hip' is ready for execution attempt",
                    evidence={"candidate": baseline_candidate},
                )
            )
        elif any(a.verdict == EvaluationVerdict.UNKNOWN for a in device_assessments):
            overall_verdict = EvaluationVerdict.UNKNOWN
            baseline_candidate = None
            reasons.append(
                EvaluationReason(
                    code="INSUFFICIENT_DATA",
                    severity=EvaluationSeverity.WARNING,
                    message="Insufficient data to guarantee baseline execution support on detected devices",
                    evidence={
                        "unknown_devices": [
                            a.device_id for a in device_assessments if a.verdict == EvaluationVerdict.UNKNOWN
                        ]
                    },
                )
            )
        else:
            overall_verdict = EvaluationVerdict.BLOCKED
            baseline_candidate = None

        capabilities = SystemCapabilities(
            amd_gpu_present=True,
            rocm_detected=rocm_detected,
            hip_detected=hip_detected,
            torch_available=torch_available,
            torch_hip_available=torch_hip_available,
            model_metadata_complete=model_metadata_complete,
            remote_code_required=remote_code_required,
            baseline_runtime_candidate=baseline_candidate,
        )

        return CapabilityReport(
            model=model,
            environment=env,
            hardware=gpus,
            device_assessments=device_assessments,
            verdict=overall_verdict,
            reasons=reasons,
            warnings=warnings,
            capabilities=capabilities,
        )
