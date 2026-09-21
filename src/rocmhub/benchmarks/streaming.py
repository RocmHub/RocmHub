"""Streaming token timestamp observer for measuring Time To First Token and ITL."""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional


class TokenTimestampStreamer:
    """Streamer callback recording monotonic nanosecond timestamps on token emission.

    Compatible with Hugging Face Transformers streamer protocol (put / end).
    """

    def __init__(
        self,
        skip_prompt: bool = True,
        clock_fn: Optional[Callable[[], int]] = None,
        sync_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize streaming token timestamp observer.

        Args:
            skip_prompt: If True, ignores the first put() call which contains prompt tokens.
            clock_fn: Callable returning monotonic nanoseconds (defaults to time.perf_counter_ns).
            sync_fn: Optional accelerator synchronization callback executed before timestamp capture.
        """
        self.skip_prompt = skip_prompt
        self.clock_fn: Callable[[], int] = clock_fn if clock_fn is not None else time.perf_counter_ns
        self.sync_fn: Optional[Callable[[], None]] = sync_fn

        self.prompt_seen: bool = False
        self.is_finished: bool = False
        self.token_timestamps_ns: List[int] = []

    def put(self, value: Any) -> None:
        """Receive newly emitted token tensor from model generation loop.

        Args:
            value: Tensor of token ids from generator.
        """
        if self.skip_prompt and not self.prompt_seen:
            self.prompt_seen = True
            return

        if self.sync_fn is not None:
            self.sync_fn()

        timestamp = self.clock_fn()
        self.token_timestamps_ns.append(timestamp)

    def end(self) -> None:
        """Signal generation completion."""
        if self.sync_fn is not None:
            self.sync_fn()
        self.is_finished = True

    def reset(self) -> None:
        """Reset internal state for another iteration."""
        self.prompt_seen = False
        self.is_finished = False
        self.token_timestamps_ns.clear()
