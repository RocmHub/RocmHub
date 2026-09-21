"""Memory tracking utility for allocator-observed GPU VRAM consumption."""

from __future__ import annotations

from typing import Any, Optional

try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    torch = None  # type: ignore[assignment]


class MemoryTracker:
    """Tracks PyTorch allocator peak memory stats for a specific GPU device.

    Important Semantic Distinction:
    This measures peak allocated memory registered by the PyTorch caching allocator
    (via torch.cuda.max_memory_allocated), NOT the total physical VRAM consumption
    of the entire operating system or GPU board.
    """

    def __init__(self, torch_module: Optional[Any] = None) -> None:
        """Initialize memory tracker.

        Args:
            torch_module: Optional injected torch module (for unit testing / mocking).
        """
        self._torch = torch_module if torch_module is not None else torch

    def is_available(self, device_id: int = 0) -> bool:
        """Check whether GPU memory tracking is available for device_id.

        Args:
            device_id: Accelerator index.

        Returns:
            True if PyTorch with CUDA/HIP runtime is available and device exists.
        """
        if self._torch is None:
            return False
        cuda = getattr(self._torch, "cuda", None)
        if cuda is None or not callable(getattr(cuda, "is_available", None)):
            return False
        if not cuda.is_available():
            return False
        count_fn = getattr(cuda, "device_count", None)
        if callable(count_fn) and device_id >= count_fn():
            return False
        return True

    def reset_peak_stats(self, device_id: int = 0) -> None:
        """Reset peak memory tracking counters before a measurement iteration.

        Args:
            device_id: Target accelerator device index.
        """
        if not self.is_available(device_id):
            return
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "reset_peak_memory_stats"):
            try:
                cuda.reset_peak_memory_stats(device_id)
            except Exception:
                # Silently ignore unsupported hardware/virtual driver calls
                pass

    def get_peak_mb(self, device_id: int = 0) -> Optional[float]:
        """Get peak allocator memory allocated since last reset in megabytes.

        Args:
            device_id: Target accelerator device index.

        Returns:
            Peak memory allocated in MB as float, or None if unavailable.
        """
        if not self.is_available(device_id):
            return None
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "max_memory_allocated"):
            try:
                peak_bytes = cuda.max_memory_allocated(device_id)
                if peak_bytes is None:
                    return None
                return float(peak_bytes) / (1024.0 * 1024.0)
            except Exception:
                return None
        return None
