# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for performance measurement."""

import json
import os
import time
from contextlib import ContextDecorator
from datetime import datetime
from pathlib import Path

import torch

from . import distributed as dist
from .logging import print_rank_0

__all__ = [
    "AccumulatingTimer",
    "Timer",
    "clear_cuda_cache",
    "format_duration",
    "get_cuda_memory_stats",
    "get_used_gpu_mem_fraction",
    "record_timing_event",
    "report_memory",
]


def clear_cuda_cache():
    """Clear the CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds for human-readable logs."""
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{secs:05.2f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{secs:05.2f}s"


def record_timing_event(event: str, seconds: float | None = None, **extra) -> None:
    """Append one timing event to ``MODELOPT_TIMING_LOG_PATH`` as JSONL.

    The log path is intentionally environment-driven so nested ModelOpt code can
    persist timings without threading script-specific arguments through every API.
    """

    log_path = os.environ.get("MODELOPT_TIMING_LOG_PATH")
    if not log_path:
        return

    payload = {
        "event": event,
        "time_unix": time.time(),
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "run_id": os.environ.get("MODELOPT_TIMING_RUN_ID"),
    }
    if seconds is not None:
        payload["seconds"] = float(seconds)
        payload["duration"] = format_duration(seconds)
    payload.update(extra)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def get_cuda_memory_stats(device=None):
    """Get memory usage of specified GPU in Bytes."""
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
    }


def get_used_gpu_mem_fraction(device="cuda:0"):
    """Get used GPU memory as a fraction of total memory.

    Args:
        device: Device identifier (default: "cuda:0")

    Returns:
        float: Fraction of GPU memory currently used (0.0 to 1.0).
               Returns 0.0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0

    free_memory, total_memory = torch.cuda.mem_get_info(device)
    return (total_memory - free_memory) / total_memory


def report_memory(name="", rank=0, device=None):
    """Simple GPU memory report."""
    memory_stats = get_cuda_memory_stats(device)
    string = name + " memory (MB)"
    for k, v in memory_stats.items():
        string += f" | {k}: {v / 2**20: .2e}"

    if dist.is_initialized():
        string = f"[Rank {dist.rank()}] " + string
        if dist.rank() == rank:
            print(string, flush=True)
    else:
        print(string, flush=True)


class Timer(ContextDecorator):
    """A Timer that can be used as a decorator as well."""

    def __init__(self, name=""):
        """Initialize Timer."""
        super().__init__()
        self.name = name
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._stop_event = torch.cuda.Event(enable_timing=True)
        self.estimated_time = 0
        self.start()

    def start(self):
        """Start the timer."""
        self._start_event.record()
        return self

    def stop(self) -> float:
        """End the timer."""
        self._stop_event.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        self.estimated_time = self._start_event.elapsed_time(self._stop_event)
        return self.estimated_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        print_rank_0(f"{self.name} took {self.estimated_time:.3e} ms")


class AccumulatingTimer(ContextDecorator):
    """A timer that accumulates time across multiple calls and works for both CUDA and non-CUDA operations."""

    # Class-level dictionary to store accumulated times by name
    _accumulated_times = {}
    _call_counts = {}
    _prefix = []

    def __init__(self, name=""):
        """Initialize AccumulatingTimer.

        Args:
            name: Name of the timer for reporting
            use_cuda: Whether to synchronize CUDA before timing
        """
        super().__init__()
        self.name = name
        self.use_cuda = torch.cuda.is_available()
        self._start_time = None

    def start(self) -> None:
        """Start the timer."""
        if self.use_cuda:
            # Synchronize CUDA before measuring start time
            torch.cuda.synchronize()
        self._start_time = time.time()

    def stop(self) -> float:
        """End the timer and return the elapsed time in milliseconds."""
        if self.use_cuda:
            # Synchronize CUDA before measuring stop time
            torch.cuda.synchronize()

        elapsed_time = (time.time() - self._start_time) * 1000  # in milliseconds

        # Update the accumulated time and call count
        name = self.name if not AccumulatingTimer._prefix else "->".join(AccumulatingTimer._prefix)
        if name not in AccumulatingTimer._accumulated_times:
            AccumulatingTimer._accumulated_times[name] = 0.0
            AccumulatingTimer._call_counts[name] = 0
        AccumulatingTimer._accumulated_times[name] += elapsed_time
        AccumulatingTimer._call_counts[name] += 1

        return elapsed_time

    @classmethod
    def get_total_time(cls, name):
        """Get the total accumulated time for a timer in milliseconds."""
        return cls._accumulated_times.get(name, 0.0)

    @classmethod
    def get_call_count(cls, name):
        """Get the number of calls for a timer."""
        return cls._call_counts.get(name, 0)

    @classmethod
    def reset(cls):
        """Reset the accumulated times and call counts."""
        cls._accumulated_times = {}
        cls._call_counts = {}

    @classmethod
    def report(cls):
        """Report the accumulated times and call counts."""
        for name, t in cls._accumulated_times.items():
            print(f"{name}: {t:0.3f} ms (avg: {t / cls._call_counts[name]:0.3f} ms)")

    def __enter__(self):
        AccumulatingTimer._prefix.append(self.name)
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        AccumulatingTimer._prefix.pop()
