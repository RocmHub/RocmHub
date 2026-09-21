"""Versioned policy representation for preflight capability evaluation."""

from __future__ import annotations

from typing import Set

from pydantic import BaseModel, ConfigDict, Field


class CapabilityPolicy(BaseModel):
    """Small, versioned capability policy for baseline preflight evaluation.

    Separation of concerns:
    - HardwareDetector & SystemObserver collect objective facts from the host.
    - CapabilityPolicy defines rule thresholds, reference architecture sets, and
      format constraints for interpreting those facts.
    - Architecture strings are dynamic; unlisted targets result in UNKNOWN
      rather than BLOCKED, ensuring forward compatibility with unreleased hardware.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(
        default="1.0.0",
        description="Policy specification version",
    )

    # Architectures recognized with standard upstream ROCm compiler / runtime support
    known_gfx_targets: Set[str] = Field(
        default_factory=lambda: {
            # CDNA 1 / 2 / 3
            "gfx908",   # Instinct MI100
            "gfx90a",   # Instinct MI210 / MI250 / MI250X
            "gfx940",   # Instinct MI300
            "gfx941",   # Instinct MI300
            "gfx942",   # Instinct MI300A / MI300X
            # RDNA 3 / 3.5
            "gfx1100",  # Radeon RX 7900 XTX / XT / GRE, PRO W7900
            "gfx1101",  # Radeon RX 7800 XT / 7700 XT, PRO W7800
            "gfx1102",  # Radeon RX 7600 / 7600 XT
            "gfx1103",  # Phoenix / Hawk Point APU
            "gfx1150",  # Strix Point
            "gfx1151",  # Strix Point
            # RDNA 2
            "gfx1030",  # Radeon RX 6800 / 6900
            "gfx1031",  # Radeon RX 6700
            "gfx1032",  # Radeon RX 6600
            # RDNA 4 (upcoming reference)
            "gfx1200",
            "gfx1201",
        },
        description="Set of known/tested AMD GFX architectures with upstream ROCm compiler support",
    )

    supported_weight_formats: Set[str] = Field(
        default_factory=lambda: {"safetensors", "pytorch_bin", "unknown"},
        description="Weights formats compatible with PyTorch + Transformers baseline loader",
    )

    unsupported_weight_formats: Set[str] = Field(
        default_factory=lambda: {"gguf"},
        description="Weights formats requiring alternative runtimes (e.g. llama.cpp) rather than HF baseline",
    )
