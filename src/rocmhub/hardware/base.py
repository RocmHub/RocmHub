"""Base protocols and contracts for hardware and environment observation."""

from __future__ import annotations

from typing import Dict, List, Protocol, Tuple, runtime_checkable

from rocmhub.core.types import HardwareSpec


@runtime_checkable
class HardwareDetector(Protocol):
    """Protocol for observing host GPU devices without policy decisions."""

    def detect_gpus(self) -> Tuple[List[HardwareSpec], Dict[str, str], List[str]]:
        """Discover and return detected GPU devices, source provenance, and non-fatal warnings.

        Returns:
            gpus: List of HardwareSpec (empty if no GPU present).
            provenance: Mapping of discovered fields to their observation source.
            warnings: Non-fatal diagnostic warnings collected during observation.
        """
        ...
