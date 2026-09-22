"""Optimization recipes and strategy definitions for ROCmHub (Phase 12)."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

from rocmhub.core.types import HardwareSpec, ModelSpec
from rocmhub.optimization.base import OptimizationStrategy


@runtime_checkable
class OptimizationRecipe(Protocol):
    """Protocol for concrete optimization strategies."""

    @property
    def strategy(self) -> OptimizationStrategy:
        """Optimization strategy enum."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name of strategy."""
        ...

    @property
    def description(self) -> str:
        """Technical description of applied transformations."""
        ...

    @property
    def precision(self) -> str:
        """Floating point precision ('bf16', 'fp16', 'fp32')."""
        ...

    @property
    def runtime_flags(self) -> Dict[str, Any]:
        """Runtime execution flags passed to runner."""
        ...

    def is_supported(
        self, model_spec: ModelSpec, hardware_spec: Optional[HardwareSpec]
    ) -> Tuple[bool, Optional[str]]:
        """Determine whether this recipe is valid for target model and hardware.

        Returns:
            Tuple of (is_supported, optional_unsupported_reason).
        """
        ...


class BF16OptimizationRecipe:
    """Strategy applying native Bfloat16 precision."""

    def __init__(self) -> None:
        self.strategy = OptimizationStrategy.BF16
        self.name = "Bfloat16 Precision"
        self.description = "Execute model in native bfloat16 format with high dynamic range."
        self.precision = "bf16"
        self.runtime_flags: Dict[str, Any] = {"torch_dtype": "bfloat16"}

    def is_supported(
        self, model_spec: ModelSpec, hardware_spec: Optional[HardwareSpec]
    ) -> Tuple[bool, Optional[str]]:
        # bfloat16 is supported on CDNA (gfx90a, gfx942) and modern RDNA3 (gfx1100).
        # On Mac / CPU, it is statically valid for configuration planning.
        return True, None


class FP16OptimizationRecipe:
    """Strategy applying native Float16 half-precision."""

    def __init__(self) -> None:
        self.strategy = OptimizationStrategy.FP16
        self.name = "Float16 Precision"
        self.description = "Execute model in standard IEEE 754 half-precision float16."
        self.precision = "fp16"
        self.runtime_flags: Dict[str, Any] = {"torch_dtype": "float16"}

    def is_supported(
        self, model_spec: ModelSpec, hardware_spec: Optional[HardwareSpec]
    ) -> Tuple[bool, Optional[str]]:
        return True, None


class FP32OptimizationRecipe:
    """Reference strategy applying full Float32 precision."""

    def __init__(self) -> None:
        self.strategy = OptimizationStrategy.FP32
        self.name = "Float32 Precision (Reference/Fallback)"
        self.description = "Execute model in full single-precision float32 for reference baseline."
        self.precision = "fp32"
        self.runtime_flags: Dict[str, Any] = {"torch_dtype": "float32"}

    def is_supported(
        self, model_spec: ModelSpec, hardware_spec: Optional[HardwareSpec]
    ) -> Tuple[bool, Optional[str]]:
        # Check if model fits in memory if hardware_spec is known
        if hardware_spec and hardware_spec.vram_total_mb and model_spec.parameter_count:
            # fp32 requires ~4 bytes per parameter
            required_mb = (model_spec.parameter_count * 4) / (1024 * 1024)
            if required_mb > hardware_spec.vram_total_mb:
                return (
                    False,
                    f"Model in FP32 requires ~{required_mb:.1f} MB VRAM, exceeding device capacity ({hardware_spec.vram_total_mb} MB)",
                )
        return True, None


class TorchCompileRecipe:
    """Strategy applying PyTorch 2 Inductor / torch.compile."""

    def __init__(self, mode: str = "reduce-overhead") -> None:
        self.strategy = OptimizationStrategy.TORCH_COMPILE
        self.name = f"PyTorch Compile ({mode})"
        self.description = f"Graph compilation via torch.compile(mode='{mode}') for fused kernel execution."
        self.precision = "fp16"
        self.runtime_flags: Dict[str, Any] = {
            "torch_compile": True,
            "compile_mode": mode,
        }

    def is_supported(
        self, model_spec: ModelSpec, hardware_spec: Optional[HardwareSpec]
    ) -> Tuple[bool, Optional[str]]:
        return True, None


class QuantizationRecipe:
    """Strategy for INT8 or FP8 post-training quantization."""

    def __init__(self, strategy: OptimizationStrategy) -> None:
        self.strategy = strategy
        self.name = f"{strategy.value.upper()} Quantization"
        self.description = f"Post-training weight quantization to {strategy.value.upper()}."
        self.precision = "fp16"  # weights loaded/quantized
        self.runtime_flags: Dict[str, Any] = {"quantization": strategy.value}

    def is_supported(
        self, model_spec: ModelSpec, hardware_spec: Optional[HardwareSpec]
    ) -> Tuple[bool, Optional[str]]:
        # Strict validation: check if required quantization libraries are present
        if self.strategy == OptimizationStrategy.INT8:
            has_bnb = importlib.util.find_spec("bitsandbytes") is not None
            has_autoawq = importlib.util.find_spec("awq") is not None
            if not has_bnb and not has_autoawq:
                return (
                    False,
                    "UNSUPPORTED: Neither 'bitsandbytes' nor 'autoawq' is installed in the current environment.",
                )
            return True, None
        elif self.strategy == OptimizationStrategy.FP8:
            # FP8 requires specific AMD CDNA3 architecture (gfx942) or ROCm >= 6.1 with PyTorch FP8 support
            if hardware_spec:
                if hardware_spec.gfx_target not in ("gfx942", "gfx940"):
                    return (
                        False,
                        f"UNSUPPORTED: FP8 hardware acceleration requires AMD Instinct CDNA3 (gfx942), got '{hardware_spec.gfx_target}'.",
                    )
            return (
                False,
                "UNSUPPORTED: AMD FP8 native kernels require ROCm 6.2+ and PyTorch FP8 transformer engine.",
            )
        return False, f"UNSUPPORTED: Strategy '{self.strategy.value}' is not yet implemented."


def get_recipe_for_strategy(strategy: OptimizationStrategy) -> OptimizationRecipe:
    """Factory returning concrete recipe instance for requested strategy."""
    if strategy == OptimizationStrategy.BF16:
        return BF16OptimizationRecipe()
    elif strategy == OptimizationStrategy.FP16:
        return FP16OptimizationRecipe()
    elif strategy == OptimizationStrategy.FP32:
        return FP32OptimizationRecipe()
    elif strategy == OptimizationStrategy.TORCH_COMPILE:
        return TorchCompileRecipe()
    elif strategy in (OptimizationStrategy.INT8, OptimizationStrategy.FP8):
        return QuantizationRecipe(strategy)
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy}")
