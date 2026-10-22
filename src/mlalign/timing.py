"""CPU-only stage timing for an alignment run; no workflow dependencies."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import os
import time

import torch


class RunTimings:
    def __init__(self):
        self.stages = {}
        self.threads = {
            "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
            "environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        }

    @contextmanager
    def stage(self, name):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        status, error = "completed", None
        print(f"[timing] {name}: started", flush=True)
        try:
            yield
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row = {
                "wall_seconds": time.perf_counter() - wall_start,
                "process_cpu_seconds": time.process_time() - cpu_start,
                "status": status,
                "error": error,
            }
            self.stages.setdefault(name, []).append(row)
            print(
                f"[timing] {name}: wall={row['wall_seconds']:.3f}s "
                f"cpu={row['process_cpu_seconds']:.3f}s status={status}",
                flush=True,
            )

    def call(self, name, function, *args, **kwargs):
        with self.stage(name):
            return function(*args, **kwargs)

    def snapshot(self):
        return {
            "schema_version": 1,
            "unit": "seconds",
            "scope": (
                "run_alignment; excludes imports, summary writing "
                "and external processes"
            ),
            "threads": deepcopy(self.threads),
            "stages": deepcopy(self.stages),
        }
