"""Benchmark harness orchestrating warmup, measurement runs, and hardware synchronization."""

from __future__ import annotations

import time
from typing import Callable, List, Optional

try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    torch = None  # type: ignore[assignment]

from rocmhub.benchmarks.base import BenchmarkConfig, BenchmarkRunMeasurement
from rocmhub.benchmarks.memory import MemoryTracker
from rocmhub.benchmarks.metrics import MetricsCalculator
from rocmhub.benchmarks.streaming import TokenTimestampStreamer
from rocmhub.core.types import BenchmarkResult, ExecutionStatus, ModelSpec
from rocmhub.runners.base import BaseRunner


class BenchmarkHarness:
    """Orchestrates timed benchmark workloads on a loaded BaseRunner.

    Responsibilities:
    - Warmup execution to trigger HIP/MIOpen kernel compilation and memory allocator warm-up.
    - Explicit device synchronization (torch.cuda.synchronize) before and after measured loops.
    - Capturing high-resolution monotonic timestamps for token streaming.
    - Tracking PyTorch allocator peak memory stats.
    - Aggregating individual run measurements into canonical BenchmarkResult.
    """

    def __init__(
        self,
        runner: BaseRunner,
        config: BenchmarkConfig,
        clock_fn: Optional[Callable[[], int]] = None,
        sync_fn: Optional[Callable[[], None]] = None,
        memory_tracker: Optional[MemoryTracker] = None,
    ) -> None:
        """Initialize benchmark harness.

        Args:
            runner: Initialized and loaded BaseRunner instance.
            config: Validated BenchmarkConfig workload parameters.
            clock_fn: Optional monotonic nanosecond timer (defaults to time.perf_counter_ns).
            sync_fn: Optional accelerator synchronization callback.
            memory_tracker: Optional MemoryTracker instance.
        """
        self.runner = runner
        self.config = config
        self.clock_fn: Callable[[], int] = clock_fn if clock_fn is not None else time.perf_counter_ns

        # Setup GPU synchronization callback
        if sync_fn is not None:
            self._sync_fn: Optional[Callable[[], None]] = sync_fn
        elif self.config.synchronize_device:
            self._sync_fn = self._create_default_sync_fn(self.config.device_id)
        else:
            self._sync_fn = None

        # Setup memory tracker
        if memory_tracker is not None:
            self.memory_tracker = memory_tracker
        elif self.config.collect_memory:
            self.memory_tracker = MemoryTracker()
        else:
            self.memory_tracker = MemoryTracker(torch_module=None)

        self._measurements: List[BenchmarkRunMeasurement] = []

    @staticmethod
    def _create_default_sync_fn(device_id: int) -> Optional[Callable[[], None]]:
        """Create a default torch.cuda.synchronize callback if torch is available."""
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                def _sync() -> None:
                    torch.cuda.synchronize(device_id)

                return _sync
            except Exception:
                return None
        return None

    @property
    def measurements(self) -> List[BenchmarkRunMeasurement]:
        """Access list of recorded run measurements."""
        return list(self._measurements)

    def _execute_single_run(
        self,
        run_index: int,
        is_warmup: bool,
    ) -> BenchmarkRunMeasurement:
        """Execute a single generation cycle with streaming timestamps and synchronization."""
        # 1. Synchronize device before measured block
        if self._sync_fn is not None:
            try:
                self._sync_fn()
            except Exception:
                pass

        # 2. Reset allocator peak memory stats
        if self.config.collect_memory:
            self.memory_tracker.reset_peak_stats(self.config.device_id)

        # 3. Streamer & Timing setup
        streamer = TokenTimestampStreamer(
            skip_prompt=True,
            clock_fn=self.clock_fn,
            sync_fn=self._sync_fn,
        )

        started_at_ns = self.clock_fn()
        error_msg: Optional[str] = None
        status = ExecutionStatus.SUCCESS
        input_tokens: Optional[int] = None
        generated_tokens: Optional[int] = None
        token_timestamps_ns: List[int] = []

        try:
            # Check if runner implements generate_stream directly
            if hasattr(self.runner, "generate_stream"):
                gen_fn = getattr(self.runner, "generate_stream")
                run_result, timestamps = gen_fn(
                    prompt=self.config.prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    streamer=streamer,
                )
                input_tokens = run_result.input_tokens
                generated_tokens = run_result.generated_tokens
                token_timestamps_ns = list(timestamps) if timestamps else list(streamer.token_timestamps_ns)
            else:
                run_result = self.runner.generate(
                    prompt=self.config.prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    streamer=streamer,
                )
                input_tokens = run_result.input_tokens
                generated_tokens = run_result.generated_tokens
                token_timestamps_ns = list(streamer.token_timestamps_ns)

            if run_result.status != ExecutionStatus.SUCCESS:
                status = ExecutionStatus.FAILED
                error_msg = run_result.error or "Runner reported execution failure"

        except Exception as exc:
            status = ExecutionStatus.FAILED
            error_msg = str(exc)

        # 4. Synchronize device after generation completion
        if self._sync_fn is not None:
            try:
                self._sync_fn()
            except Exception:
                pass

        finished_at_ns = self.clock_fn()

        # 5. Collect peak allocator memory
        peak_mb = (
            self.memory_tracker.get_peak_mb(self.config.device_id)
            if self.config.collect_memory
            else None
        )

        return MetricsCalculator.calculate_run_measurement(
            run_index=run_index,
            is_warmup=is_warmup,
            started_at_ns=started_at_ns,
            finished_at_ns=finished_at_ns,
            token_timestamps_ns=token_timestamps_ns,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            peak_vram_used_mb=peak_mb,
            status=status,
            error=error_msg,
        )

    def run(self, model: Optional[ModelSpec] = None) -> BenchmarkResult:
        """Execute complete benchmark suite: warmups followed by measurement iterations.

        Args:
            model: Optional ModelSpec for populating result metadata.

        Returns:
            Aggregated BenchmarkResult.
        """
        self._measurements.clear()

        # 1. Warmup cycles (strictly excluded from metric calculations)
        for w_idx in range(self.config.warmup_runs):
            warmup_m = self._execute_single_run(run_index=w_idx, is_warmup=True)
            self._measurements.append(warmup_m)

        # 2. Measurement cycles
        for m_idx in range(self.config.measurement_runs):
            measurement_m = self._execute_single_run(run_index=m_idx, is_warmup=False)
            self._measurements.append(measurement_m)
            if measurement_m.status == ExecutionStatus.FAILED:
                # Stop on failure and preserve evidence
                break

        # 3. Aggregate results
        model_id = model.model_id if model else None
        model_revision = model.commit_sha if model else None

        return MetricsCalculator.aggregate(
            config=self.config,
            measurements=self._measurements,
            model_id=model_id,
            model_revision=model_revision,
            runtime_name=self.runner.runner_name,
        )
