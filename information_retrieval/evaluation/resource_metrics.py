"""
Lightweight CPU / RAM / GPU resource sampling for individual queries.

Used as a context manager around the heavy parts of a query so we can report
back to the UI roughly how expensive the run was. All values are best-effort;
if a backend (psutil, torch.cuda) isn't available, the corresponding fields
are simply omitted.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

try:
    import psutil  # type: ignore
    _PROC = psutil.Process(os.getpid())
    # Prime the cpu_percent counter; the first call always returns 0.0.
    _PROC.cpu_percent(interval=None)
    _PSUTIL_OK = True
except Exception:  # pragma: no cover
    _PROC = None
    _PSUTIL_OK = False


def _gpu_snapshot() -> Optional[Dict[str, float]]:
    """Return GPU memory usage in MB, or None if torch/CUDA isn't available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device()
        return {
            "gpu_mem_used_mb":   torch.cuda.memory_allocated(idx) / (1024 * 1024),
            "gpu_mem_peak_mb":   torch.cuda.max_memory_allocated(idx) / (1024 * 1024),
            "gpu_name":          torch.cuda.get_device_name(idx),
        }
    except Exception:
        return None


class ResourceSampler:
    """Context manager that captures resource usage between enter and exit."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.start_rss: Optional[float] = None
        self.end_rss: Optional[float] = None
        self.cpu_pct: Optional[float] = None
        self.gpu_start: Optional[Dict[str, float]] = None
        self.gpu_end: Optional[Dict[str, float]] = None

    def __enter__(self) -> "ResourceSampler":
        self.start_time = time.perf_counter()
        if _PSUTIL_OK and _PROC is not None:
            try:
                self.start_rss = _PROC.memory_info().rss / (1024 * 1024)
                _PROC.cpu_percent(interval=None)  # reset
            except Exception:
                self.start_rss = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self.gpu_start = _gpu_snapshot()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end_time = time.perf_counter()
        if _PSUTIL_OK and _PROC is not None:
            try:
                self.end_rss = _PROC.memory_info().rss / (1024 * 1024)
                # cpu_percent without an interval returns the % since the last call,
                # which we primed at __enter__.
                self.cpu_pct = _PROC.cpu_percent(interval=None)
            except Exception:
                self.end_rss = None
                self.cpu_pct = None
        self.gpu_end = _gpu_snapshot()

    def to_dict(self) -> Dict:
        out: Dict = {
            "elapsed_ms": round((self.end_time - self.start_time) * 1000, 2),
        }
        if self.cpu_pct is not None:
            out["cpu_percent"] = round(self.cpu_pct, 2)
        if self.start_rss is not None and self.end_rss is not None:
            out["ram_used_mb"]  = round(self.end_rss, 2)
            out["ram_delta_mb"] = round(self.end_rss - self.start_rss, 2)
        if self.gpu_end:
            out["gpu_mem_mb"]      = round(self.gpu_end.get("gpu_mem_used_mb", 0.0), 2)
            out["gpu_mem_peak_mb"] = round(self.gpu_end.get("gpu_mem_peak_mb", 0.0), 2)
            if "gpu_name" in self.gpu_end:
                out["gpu_name"] = self.gpu_end["gpu_name"]
        else:
            out["gpu_available"] = False
        return out
