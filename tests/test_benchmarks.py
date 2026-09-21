"""Unit and statistical test suite for ROCmHub Benchmark Harness and Metrics.

Verifies:
- Accurate TTFT (request start -> first generated token)
- Accurate ITL (strictly consecutive token intervals, TTFT excluded)
- When generated_tokens < 2, ITL is strictly None
- Deterministic linear-interpolation percentiles (p50, p90, p99, mean)
- Throughput calculation (generated tokens / end-to-end duration)
- Warmup exclusion from summary metrics
- Peak allocator memory semantics (max across runs, MB conversion, None when unavailable)
- Device synchronization calls
- Partial failures marking overall benchmark FAILED with None metrics
- Preflight non-READY blocking (NO_ACCELERATOR, BLOCKED, UNKNOWN) with zero downloads
- BenchmarkConfig bounds validation
- CLI human and JSON outputs
"""

from __future__ import annotations

import json
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from rocmhub.benchmarks.base import BenchmarkConfig
from rocmhub.benchmarks.harness import BenchmarkHarness
from rocmhub.benchmarks.memory import MemoryTracker
from rocmhub.benchmarks.metrics import MetricsCalculator, compute_median, compute_percentile
from rocmhub.cli.main import main
from rocmhub.core.errors import InvalidBenchmarkConfigError
from rocmhub.core.types import (
    BenchmarkResult,
    CapabilityReport,
    EvaluationVerdict,
    ExecutionStatus,
    ModelSpec,
    RunResult,
    SystemCapabilities,
)


class FakeMockRunner:
    """Mock runner allowing deterministic injection of token emission timestamps."""

    def __init__(
        self,
        token_timestamps_ns_per_run: Optional[List[List[int]]] = None,
        tokens_count: int = 4,
        should_fail_on_run: Optional[int] = None,
    ) -> None:
        self.runner_name = "pytorch_transformers_hip"
        self._timestamps_queue = list(token_timestamps_ns_per_run or [])
        self._tokens_count = tokens_count
        self._should_fail_on_run = should_fail_on_run
        self._call_count = 0
        self.unloaded = False

    def supports(self, capability_report: CapabilityReport, device_id: int) -> bool:
        return True

    def load(self, model: ModelSpec, device_id: int, precision: str = "fp16") -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 16, **kwargs: Any) -> RunResult:
        streamer = kwargs.get("streamer")
        run_idx = self._call_count
        self._call_count += 1

        if self._should_fail_on_run is not None and run_idx == self._should_fail_on_run:
            return RunResult(
                status=ExecutionStatus.FAILED,
                runtime_name=self.runner_name,
                model_id="test/model",
                model_revision="0123456789abcdef0123456789abcdef01234567",
                device_id=0,
                precision="fp16",
                prompt=prompt,
                error="Simulated generation device crash",
            )

        offsets = (
            self._timestamps_queue.pop(0)
            if self._timestamps_queue
            else [100_000_000, 130_000_000, 170_000_000, 220_000_000]
        )

        if streamer is not None:
            # Simulate Transformers streamer call: first prompt, then generated tokens
            streamer.put("prompt_tensor")
            base_time = streamer.clock_fn() if hasattr(streamer, "clock_fn") else 0
            for offset in offsets:
                streamer.token_timestamps_ns.append(base_time + offset)
            streamer.end()

        return RunResult(
            status=ExecutionStatus.SUCCESS,
            runtime_name=self.runner_name,
            model_id="test/model",
            model_revision="0123456789abcdef0123456789abcdef01234567",
            device_id=0,
            precision="fp16",
            prompt=prompt,
            generated_text="test output",
            input_tokens=10,
            generated_tokens=len(offsets),
        )

    def unload(self) -> None:
        self.unloaded = True


# =========================================================================
# 1. Exact Synthetic Timestamps & Metric Definitions
# =========================================================================

class TestSyntheticTimestampsAndMetrics:
    """Mathematical verification using synthetic monotonic timestamps."""

    def test_exact_synthetic_timestamps_mathematical_verification(self) -> None:
        """Prompt #17 exact check:
        request_start = 0 ms (0 ns)
        token1 = 100 ms (100_000_000 ns)
        token2 = 130 ms (130_000_000 ns)
        token3 = 170 ms (170_000_000 ns)
        token4 = 220 ms (220_000_000 ns)
        finished = 230 ms (230_000_000 ns)

        Expectations:
        - TTFT = 100.0 ms
        - ITL samples: [30.0, 40.0, 50.0] ms
        - Total Latency = 230.0 ms
        - Throughput = 4 tokens / 0.220 s = 18.1818... tokens/s
        """
        started_at_ns = 0
        token_timestamps_ns = [100_000_000, 130_000_000, 170_000_000, 220_000_000]
        finished_at_ns = 230_000_000

        meas = MetricsCalculator.calculate_run_measurement(
            run_index=0,
            is_warmup=False,
            started_at_ns=started_at_ns,
            finished_at_ns=finished_at_ns,
            token_timestamps_ns=token_timestamps_ns,
            input_tokens=10,
            generated_tokens=4,
            peak_vram_used_mb=1024.0,
            status=ExecutionStatus.SUCCESS,
        )

        # TTFT: request start -> first generated token
        assert meas.ttft_ms == pytest.approx(100.0, rel=1e-5)

        # ITL: deltas between consecutive generated tokens (TTFT excluded)
        assert len(meas.inter_token_latencies_ms) == 3
        assert meas.inter_token_latencies_ms[0] == pytest.approx(30.0, rel=1e-5)  # 130 - 100
        assert meas.inter_token_latencies_ms[1] == pytest.approx(40.0, rel=1e-5)  # 170 - 130
        assert meas.inter_token_latencies_ms[2] == pytest.approx(50.0, rel=1e-5)  # 220 - 170

        # Total latency
        assert meas.total_latency_ms == pytest.approx(230.0, rel=1e-5)

    def test_ttft_is_not_total_latency(self) -> None:
        """Verify TTFT is strictly differentiated from total generation latency."""
        meas = MetricsCalculator.calculate_run_measurement(
            run_index=0,
            is_warmup=False,
            started_at_ns=10_000_000,
            finished_at_ns=500_000_000,
            token_timestamps_ns=[50_000_000, 150_000_000, 480_000_000],
            input_tokens=5,
            generated_tokens=3,
            peak_vram_used_mb=None,
        )
        assert meas.ttft_ms == pytest.approx(40.0, rel=1e-5)
        assert meas.total_latency_ms == pytest.approx(490.0, rel=1e-5)
        assert meas.ttft_ms < meas.total_latency_ms

    def test_ttft_excluded_from_itl(self) -> None:
        """Verify TTFT (e.g. 100ms) is never included in the ITL series."""
        meas = MetricsCalculator.calculate_run_measurement(
            run_index=0,
            is_warmup=False,
            started_at_ns=0,
            finished_at_ns=300_000_000,
            token_timestamps_ns=[100_000_000, 130_000_000, 170_000_000],
            input_tokens=8,
            generated_tokens=3,
            peak_vram_used_mb=None,
        )
        assert meas.ttft_ms == 100.0
        assert meas.inter_token_latencies_ms == [30.0, 40.0]
        assert 100.0 not in meas.inter_token_latencies_ms

    def test_single_token_itl_is_none(self) -> None:
        """When generated_tokens < 2, ITL is undefined (None) rather than 0."""
        meas = MetricsCalculator.calculate_run_measurement(
            run_index=0,
            is_warmup=False,
            started_at_ns=0,
            finished_at_ns=150_000_000,
            token_timestamps_ns=[120_000_000],
            input_tokens=10,
            generated_tokens=1,
            peak_vram_used_mb=None,
        )
        assert meas.ttft_ms == 120.0
        assert meas.inter_token_latencies_ms == []

        config = BenchmarkConfig(warmup_runs=0, measurement_runs=1, max_new_tokens=1)
        res = MetricsCalculator.aggregate(config, [meas])
        assert res.status == ExecutionStatus.SUCCESS
        assert res.ttft_ms == 120.0
        assert res.itl_ms_mean is None
        assert res.itl_ms_p50 is None
        assert res.itl_ms_p90 is None
        assert res.itl_ms_p99 is None

    def test_deterministic_percentiles(self) -> None:
        """Verify exact linear-interpolation percentile mathematical values."""
        # For [10, 20, 30, 40, 50]
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert compute_percentile(data, 0.0) == 10.0
        assert compute_percentile(data, 50.0) == 30.0
        assert compute_median(data) == 30.0
        assert compute_percentile(data, 100.0) == 50.0
        assert compute_percentile(data, 90.0) == pytest.approx(46.0)
        assert compute_percentile(data, 25.0) == pytest.approx(20.0)
        assert compute_percentile(data, 75.0) == pytest.approx(40.0)

        # Single element
        assert compute_percentile([42.5], 50.0) == 42.5
        assert compute_percentile([42.5], 99.0) == 42.5

        # Error cases
        with pytest.raises(ValueError, match="empty sequence"):
            compute_percentile([], 50.0)
        with pytest.raises(ValueError, match="between 0.0 and 100.0"):
            compute_percentile([1.0, 2.0], -1.0)
        with pytest.raises(ValueError, match="between 0.0 and 100.0"):
            compute_percentile([1.0, 2.0], 101.0)

    def test_throughput_calculation(self) -> None:
        """Verify throughput = generated_tokens / (last_token - start_time in seconds)."""
        started = 0
        # 4 tokens, last token at 200_000_000 ns = 0.200 s
        tokens = [50_000_000, 100_000_000, 150_000_000, 200_000_000]
        finished = 210_000_000

        meas = MetricsCalculator.calculate_run_measurement(
            run_index=0,
            is_warmup=False,
            started_at_ns=started,
            finished_at_ns=finished,
            token_timestamps_ns=tokens,
            input_tokens=10,
            generated_tokens=4,
            peak_vram_used_mb=500.0,
        )

        config = BenchmarkConfig(warmup_runs=0, measurement_runs=1)
        res = MetricsCalculator.aggregate(config, [meas])
        # 4 / 0.2 = 20.0 tokens/s
        assert res.throughput_tokens_per_sec == pytest.approx(20.0, rel=1e-5)


# =========================================================================
# 2. Warmup, Multi-Run Aggregation & Peak Memory
# =========================================================================

class TestHarnessExecutionAndAggregation:
    """Verification of warmup isolation, aggregation, memory, and sync."""

    def test_warmups_excluded_from_metrics(self) -> None:
        """Warmup run (e.g. cold kernel compilation taking 5000ms) is excluded from final stats."""
        warmup_meas = MetricsCalculator.calculate_run_measurement(
            run_index=0,
            is_warmup=True,
            started_at_ns=0,
            finished_at_ns=5_000_000_000,
            token_timestamps_ns=[3_000_000_000, 4_000_000_000],
            input_tokens=10,
            generated_tokens=2,
            peak_vram_used_mb=2000.0,
        )
        meas1 = MetricsCalculator.calculate_run_measurement(
            run_index=0,
            is_warmup=False,
            started_at_ns=0,
            finished_at_ns=100_000_000,
            token_timestamps_ns=[40_000_000, 80_000_000],
            input_tokens=10,
            generated_tokens=2,
            peak_vram_used_mb=1000.0,
        )
        meas2 = MetricsCalculator.calculate_run_measurement(
            run_index=1,
            is_warmup=False,
            started_at_ns=0,
            finished_at_ns=100_000_000,
            token_timestamps_ns=[50_000_000, 90_000_000],
            input_tokens=10,
            generated_tokens=2,
            peak_vram_used_mb=1100.0,
        )

        config = BenchmarkConfig(warmup_runs=1, measurement_runs=2)
        res = MetricsCalculator.aggregate(config, [warmup_meas, meas1, meas2])

        assert res.warmup_runs == 1
        assert res.measurement_runs_completed == 2
        # Median TTFT between 40ms and 50ms is 45ms (NOT 3000ms from warmup)
        assert res.ttft_ms == pytest.approx(45.0, rel=1e-5)
        # Peak memory is max across measurement runs = 1100.0 (NOT 2000.0 from warmup)
        assert res.peak_vram_used_mb == pytest.approx(1100.0, rel=1e-5)

    def test_peak_allocator_memory_tracking(self) -> None:
        """MemoryTracker correctly resets and captures peak memory in MB."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        # 1 GB in bytes = 1073741824 bytes -> 1024 MB
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 1024

        tracker = MemoryTracker(torch_module=mock_torch)
        assert tracker.is_available(0) is True

        tracker.reset_peak_stats(0)
        mock_torch.cuda.reset_peak_memory_stats.assert_called_once_with(0)

        peak_mb = tracker.get_peak_mb(0)
        assert peak_mb == pytest.approx(1024.0, rel=1e-5)

    def test_memory_unavailable_returns_none(self) -> None:
        """When PyTorch or CUDA/HIP is unavailable, memory tracker returns None."""
        tracker = MemoryTracker(torch_module=None)
        assert tracker.is_available(0) is False
        assert tracker.get_peak_mb(0) is None

    def test_device_synchronization_invoked(self) -> None:
        """Harness invokes sync_fn before and after each measured generation."""
        sync_mock = MagicMock()
        fake_runner = FakeMockRunner()
        config = BenchmarkConfig(warmup_runs=1, measurement_runs=2, synchronize_device=True)

        harness = BenchmarkHarness(
            runner=fake_runner,
            config=config,
            sync_fn=sync_mock,
        )
        res = harness.run()

        assert res.status == ExecutionStatus.SUCCESS
        # 3 runs total (1 warmup + 2 measurement) * 2 calls per run = 6 sync calls
        assert sync_mock.call_count >= 6

    def test_partial_run_failure_marks_overall_benchmark_failed(self) -> None:
        """If 1 of 5 runs fails, entire benchmark is marked FAILED and metrics remain None."""
        # Setup runner that fails on run index 3 (warmup=0, runs=1,2,3 -> fail)
        runner = FakeMockRunner(should_fail_on_run=2)
        config = BenchmarkConfig(warmup_runs=0, measurement_runs=5)

        harness = BenchmarkHarness(runner=runner, config=config)
        res = harness.run()

        assert res.status == ExecutionStatus.FAILED
        assert res.failed_runs == 1
        assert res.measurement_runs_completed == 2
        assert res.measurement_runs_requested == 5
        assert "Simulated generation device crash" in str(res.error_message)

        # Performance metrics MUST be strictly None on FAILED benchmarks
        assert res.ttft_ms is None
        assert res.itl_ms_mean is None
        assert res.throughput_tokens_per_sec is None
        assert res.peak_vram_used_mb is None

    def test_all_runs_success(self) -> None:
        """Successful runs aggregate median TTFT, ITL percentiles, and throughput."""
        # 3 runs with timestamps:
        # Run 1: TTFT=100, ITL=[30, 40]
        # Run 2: TTFT=110, ITL=[25, 35]
        # Run 3: TTFT=90,  ITL=[35, 45]
        q = [
            [100_000_000, 130_000_000, 170_000_000],
            [110_000_000, 135_000_000, 170_000_000],
            [90_000_000, 125_000_000, 170_000_000],
        ]
        runner = FakeMockRunner(token_timestamps_ns_per_run=q)
        config = BenchmarkConfig(warmup_runs=0, measurement_runs=3)

        sim_time = [1_000_000_000]

        def sim_clock() -> int:
            return sim_time[0]

        harness = BenchmarkHarness(runner=runner, config=config, clock_fn=sim_clock)
        res = harness.run()

        assert res.status == ExecutionStatus.SUCCESS
        assert res.measurement_runs_completed == 3
        # TTFT: [90, 100, 110] -> median = 100.0
        assert res.ttft_ms == pytest.approx(100.0)
        # All ITL samples: [30, 40, 25, 35, 35, 45]
        # Sorted: [25, 30, 35, 35, 40, 45], mean = 210/6 = 35.0
        assert res.itl_ms_mean == pytest.approx(35.0)
        assert res.itl_ms_p50 == pytest.approx(35.0)
        assert res.throughput_tokens_per_sec is not None
        assert res.throughput_tokens_per_sec > 0.0


# =========================================================================
# 3. Config Validation & Schema Integrity
# =========================================================================

class TestBenchmarkConfigAndIntegrity:
    """Validation of input constraints and Pydantic integrity."""

    def test_benchmark_config_bounds(self) -> None:
        with pytest.raises(InvalidBenchmarkConfigError, match="warmup_runs must be non-negative"):
            BenchmarkConfig(warmup_runs=-1)

        with pytest.raises(InvalidBenchmarkConfigError, match="measurement_runs must be at least 1"):
            BenchmarkConfig(measurement_runs=0)

        with pytest.raises(InvalidBenchmarkConfigError, match="max_new_tokens must be strictly positive"):
            BenchmarkConfig(max_new_tokens=0)

        with pytest.raises(InvalidBenchmarkConfigError, match="device_id must be non-negative"):
            BenchmarkConfig(device_id=-1)

        with pytest.raises(InvalidBenchmarkConfigError, match="Unsupported precision"):
            BenchmarkConfig(precision="int8")

    def test_not_measured_or_skipped_cannot_contain_metrics(self) -> None:
        """Diagnostic integrity: status SKIPPED / NOT_MEASURED cannot contain metrics."""
        with pytest.raises(ValueError, match="Diagnostic / non-measured results must not contain benchmark metrics"):
            BenchmarkResult(
                status=ExecutionStatus.SKIPPED,
                ttft_ms=15.0,  # Forbidden
            )

        with pytest.raises(ValueError, match="Diagnostic / non-measured results must not contain benchmark metrics"):
            BenchmarkResult(
                status=ExecutionStatus.NOT_MEASURED,
                peak_vram_used_mb=1024.0,  # Forbidden
            )


# =========================================================================
# 4. CLI Benchmark Command & Preflight Gating
# =========================================================================

class TestCliBenchmarkCommand:
    """Verification of CLI rocmhub benchmark behavior and preflight gates."""

    @pytest.fixture
    def mock_model_spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            requested_revision="main",
            commit_sha="7ae557604adf67be50417f59c2c2f167def9a775",
            architecture="Qwen2ForCausalLM",
            parameter_count=494_032_768,
            context_length=32768,
            default_dtype="bfloat16",
            weights_format="safetensors",
        )

    def test_cli_benchmark_stops_on_no_accelerator_without_loading_or_downloading(
        self,
        mock_model_spec: ModelSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """On Mac without GPU: benchmark halts with code 2 (NO_ACCELERATOR) without downloading weights."""
        with patch("rocmhub.cli.main.ModelInspector.inspect", return_value=mock_model_spec), \
             patch("rocmhub.cli.main.HuggingFaceRunner.load") as mock_load:

            exit_code = main(["benchmark", "Qwen/Qwen2.5-0.5B-Instruct"])

            # Must exit with code 2
            assert exit_code == 2

            # Model loading must NEVER have been called
            mock_load.assert_not_called()

            captured = capsys.readouterr()
            assert "Preflight Check: NO_ACCELERATOR" in captured.out
            assert "Model weights were NOT downloaded and benchmark was NOT executed." in captured.out

    def test_cli_benchmark_stops_on_blocked_preflight(
        self,
        mock_model_spec: ModelSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When preflight verdict is BLOCKED, exits with code 3 without loading."""
        from rocmhub.core.types import EnvironmentSpec

        mock_blocked_report = CapabilityReport(
            model=mock_model_spec,
            environment=EnvironmentSpec(
                os="Linux",
                python_version="3.11",
                torch_version="2.4.0",
                torch_hip_available=False,
            ),
            hardware=[],
            verdict=EvaluationVerdict.BLOCKED,
            reasons=[],
            capabilities=SystemCapabilities(),
        )

        with patch("rocmhub.cli.main.ModelInspector.inspect", return_value=mock_model_spec), \
             patch("rocmhub.cli.main.CapabilityEvaluator.evaluate", return_value=mock_blocked_report), \
             patch("rocmhub.cli.main.HuggingFaceRunner.load") as mock_load:

            exit_code = main(["benchmark", "Qwen/Qwen2.5-0.5B-Instruct"])
            assert exit_code == 3
            mock_load.assert_not_called()

            captured = capsys.readouterr()
            assert "Preflight Check: BLOCKED" in captured.out

    def test_cli_benchmark_json_output_on_skipped(
        self,
        mock_model_spec: ModelSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """CLI --json outputs valid BenchmarkResult with SKIPPED status and null metrics."""
        with patch("rocmhub.cli.main.ModelInspector.inspect", return_value=mock_model_spec):
            exit_code = main(["benchmark", "Qwen/Qwen2.5-0.5B-Instruct", "--json"])
            assert exit_code == 2

            captured = capsys.readouterr()
            res_dict = json.loads(captured.out)

            assert res_dict["status"] == "SKIPPED"
            assert res_dict["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
            assert res_dict["model_revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
            assert res_dict["ttft_ms"] is None
            assert res_dict["itl_ms_mean"] is None
            assert res_dict["throughput_tokens_per_sec"] is None
            assert res_dict["peak_vram_used_mb"] is None

    def test_cli_benchmark_ready_executes_harness_and_unloads(
        self,
        mock_model_spec: ModelSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When preflight verdict is READY, harness executes and runner is unloaded."""
        from rocmhub.core.types import EnvironmentSpec, HardwareSpec

        mock_ready_report = CapabilityReport(
            model=mock_model_spec,
            environment=EnvironmentSpec(
                os="Linux",
                python_version="3.11",
                rocm_version="6.2.0",
                hip_version="6.2.4",
                torch_version="2.4.0+rocm6.2",
                torch_hip_available=True,
            ),
            hardware=[
                HardwareSpec(
                    gpu_present=True,
                    gpu_vendor="AMD",
                    device_id=0,
                    device_name="AMD Radeon RX 7900 XTX",
                    family="Radeon",
                    gfx_target="gfx1100",
                    vram_total_mb=24576,
                    compute_units=96,
                )
            ],
            verdict=EvaluationVerdict.READY,
            reasons=[],
            capabilities=SystemCapabilities(
                amd_gpu_present=True,
                rocm_detected=True,
                hip_detected=True,
                torch_available=True,
                torch_hip_available=True,
                model_metadata_complete=True,
                remote_code_required=False,
                baseline_runtime_candidate="pytorch_transformers_hip",
            ),
        )

        mock_benchmark_res = BenchmarkResult(
            status=ExecutionStatus.SUCCESS,
            model_id=mock_model_spec.model_id,
            model_revision=mock_model_spec.commit_sha,
            device_id=0,
            runtime_name="pytorch_transformers_hip",
            precision="fp16",
            warmup_runs=2,
            measurement_runs_requested=5,
            measurement_runs_completed=5,
            failed_runs=0,
            ttft_ms=18.5,
            itl_ms_mean=8.2,
            itl_ms_p50=8.0,
            itl_ms_p90=8.8,
            itl_ms_p99=10.1,
            throughput_tokens_per_sec=121.9,
            peak_vram_used_mb=1850.5,
            total_latency_ms=149.7,
            generated_tokens_count=16,
        )

        with patch("rocmhub.cli.main.ModelInspector.inspect", return_value=mock_model_spec), \
             patch("rocmhub.cli.main.CapabilityEvaluator.evaluate", return_value=mock_ready_report), \
             patch("rocmhub.cli.main.HuggingFaceRunner.load") as mock_load, \
             patch("rocmhub.cli.main.BenchmarkHarness.run", return_value=mock_benchmark_res) as mock_run, \
             patch("rocmhub.cli.main.HuggingFaceRunner.unload") as mock_unload:

            exit_code = main(["benchmark", "Qwen/Qwen2.5-0.5B-Instruct", "--warmup-runs", "2", "--runs", "5"])
            assert exit_code == 0
            mock_load.assert_called_once()
            mock_run.assert_called_once()
            mock_unload.assert_called_once()

            captured = capsys.readouterr()
            assert "Benchmark Result:        SUCCESS" in captured.out
            assert "TTFT (median):           18.50 ms" in captured.out
            assert "ITL (p50):               8.00 ms" in captured.out
            assert "Throughput:              121.90 tokens/sec" in captured.out
            assert "Peak Allocator Memory:   1,850.50 MB" in captured.out

    def test_cli_benchmark_stops_on_unknown_preflight(
        self,
        mock_model_spec: ModelSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When preflight verdict is UNKNOWN, exits with code 4 without loading."""
        from rocmhub.core.types import EnvironmentSpec

        mock_unknown_report = CapabilityReport(
            model=mock_model_spec,
            environment=EnvironmentSpec(
                os="Linux",
                python_version="3.11",
                torch_version="2.4.0",
                torch_hip_available=True,
            ),
            hardware=[],
            verdict=EvaluationVerdict.UNKNOWN,
            reasons=[],
            capabilities=SystemCapabilities(),
        )

        with patch("rocmhub.cli.main.ModelInspector.inspect", return_value=mock_model_spec), \
             patch("rocmhub.cli.main.CapabilityEvaluator.evaluate", return_value=mock_unknown_report), \
             patch("rocmhub.cli.main.HuggingFaceRunner.load") as mock_load:

            exit_code = main(["benchmark", "Qwen/Qwen2.5-0.5B-Instruct"])
            assert exit_code == 4
            mock_load.assert_not_called()

            captured = capsys.readouterr()
            assert "Preflight Check: UNKNOWN" in captured.out

    def test_cli_benchmark_missing_model_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Invoking rocmhub benchmark with no model argument fails with exit code 1."""
        exit_code = main(["benchmark"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "model_id must be provided" in captured.err
