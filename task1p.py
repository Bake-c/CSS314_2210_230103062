"""Task 1 - Row-major vs. column-major parallel cache traversal.

8,000 x 8,000 float64 matrix (C order, 512 MB), summed in parallel by
physical-core-many threads in two modes:
  Mode A: workers own blocks of ROWS    and walk each row linearly (contiguous).
  Mode B: workers own blocks of COLUMNS and walk down each column (stride = 64,000 B,
          every element is on a different cache line).
Both modes make the same number of numpy calls on the same number of elements, so the
difference comes from memory access order. numpy releases the GIL inside reductions.

Usage: python task1.py [workers]     (default: number of physical cores)
"""
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103062
SIZE = 8000
REPEATS = 3


def physical_cores():
    """Physical core count; falls back to logical cores if it cannot be detected."""
    logical = os.cpu_count() or 1
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    try:
        if sys.platform.startswith("win"):
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            if out.isdigit():
                return min(int(out), logical)
        elif sys.platform.startswith("linux"):
            out = subprocess.run(["lscpu", "-p=CORE,SOCKET"], capture_output=True,
                                 text=True, timeout=10).stdout
            cores = {l.strip() for l in out.splitlines() if l.strip() and not l.startswith("#")}
            if cores:
                return min(len(cores), logical)
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            if out.isdigit():
                return min(int(out), logical)
    except Exception:
        pass
    return logical


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def rows_worker(m, lo, hi):
    """Mode A: row-major. Each row is a contiguous 64 KB run of memory."""
    total = 0.0
    for i in range(lo, hi):
        total += float(m[i, :].sum())
    return total


def cols_worker(m, lo, hi):
    """Mode B: column-major. Each column steps 64,000 bytes between elements."""
    total = 0.0
    for j in range(lo, hi):
        total += float(m[:, j].sum())
    return total


def run_mode(fn, m, workers):
    bounds = np.linspace(0, m.shape[0], workers + 1, dtype=np.int64)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, m, int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]
        total = sum(f.result() for f in futs)
    return (time.perf_counter() - t0) * 1000.0, total


def bench(fn, m, workers):
    run_mode(fn, m, workers)  # warm-up
    times, total = [], 0.0
    for _ in range(REPEATS):
        ms, total = run_mode(fn, m, workers)
        times.append(ms)
    return statistics.median(times), total


def main():
    logical = os.cpu_count() or 1
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else physical_cores()

    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"CPU model        : {cpu_model()}")
    print(f"Logical cores    : {logical}")
    print(f"Physical cores   : {physical_cores()}")
    print(f"Worker threads   : {workers} (physical cores)")
    print(f"Matrix           : {SIZE} x {SIZE} float64 ({SIZE * SIZE * 8 / 1e6:.0f} MB), C (row-major) layout")
    print(f"Timing           : median of {REPEATS} runs after warm-up\n")

    m = np.random.default_rng(STUDENT_ID).random((SIZE, SIZE), dtype=np.float64)

    t_a, sum_a = bench(rows_worker, m, workers)
    t_b, sum_b = bench(cols_worker, m, workers)

    assert abs(sum_a - sum_b) / abs(sum_a) < 1e-9, "modes disagree on the sum"

    print(f"Matrix sum       : {sum_a:.6f}")
    print(f"Mode A (row-major, linear)    : {t_a:10.2f} ms")
    print(f"Mode B (column-major, strided): {t_b:10.2f} ms")
    print(f"Cache penalty factor (B / A)  : {t_b / t_a:10.2f}x")


if __name__ == "__main__":
    main()
