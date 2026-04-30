"""Runtime limits for native numeric libraries and parallel workers."""

from __future__ import annotations

import os

DEFAULT_MAX_NUMERIC_THREADS = 24
DEFAULT_BLAS_THREADS = 1


def configure_numeric_threads() -> None:
    """Set conservative defaults before NumPy/SciPy native libraries load."""
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        _set_thread_env_default(name)


def max_parallel_jobs() -> int:
    """Return the maximum Python worker threads to use around BLAS-backed code."""
    return _positive_int_env("EYEFLOW_MAX_PARALLEL_JOBS") or DEFAULT_MAX_NUMERIC_THREADS


def cap_parallel_jobs(requested_jobs: int) -> int:
    """Cap a requested worker count to avoid OpenBLAS thread metadata warnings."""
    if requested_jobs < 1:
        return max_parallel_jobs()
    return max(1, min(requested_jobs, max_parallel_jobs()))


def _positive_int_env(name: str) -> int | None:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return None
    return value if value > 0 else None


def _set_thread_env_default(name: str) -> None:
    value = _positive_int_env(name)
    if value is None:
        os.environ[name] = str(DEFAULT_BLAS_THREADS)
        return
    if value > DEFAULT_MAX_NUMERIC_THREADS:
        os.environ[name] = str(DEFAULT_MAX_NUMERIC_THREADS)
