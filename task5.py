"""Task 5 - The overhead crossover point.

Benchmark y[i] = a * x[i] + b, serial vs. a thread pool using all logical
cores, and report the range of N where parallel flips from slower to faster.
numpy releases the GIL inside its loops, so threads can overlap on large arrays.
"""
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

A, B = 2.5, 1.25
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
WORKERS = os.cpu_count() or 1
BATCHES = 7
TARGET_BATCH_S = 0.03  # aim for ~30 ms per timing batch


def serial_op(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def chunk_op(x, y, lo, hi):
    np.multiply(x[lo:hi], A, out=y[lo:hi])
    np.add(y[lo:hi], B, out=y[lo:hi])


def make_parallel(pool, workers):
    def parallel_op(x, y):
        n = len(x)
        step = -(-n // workers)  # ceil division
        futures = [pool.submit(chunk_op, x, y, lo, min(lo + step, n))
                   for lo in range(0, n, step)]
        for f in futures:
            f.result()
    return parallel_op


def bench(fn, x, y):
    """Median time per call in seconds, with adaptive repeat count."""
    for _ in range(3):
        fn(x, y)  # warm-up
    t0 = time.perf_counter()
    fn(x, y)
    est = max(time.perf_counter() - t0, 1e-7)
    loops = max(1, int(TARGET_BATCH_S / est))
    times = []
    for _ in range(BATCHES):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn(x, y)
        times.append((time.perf_counter() - t0) / loops)
    return statistics.median(times)


def measure(n, par_op):
    rng = np.random.default_rng(0)
    x = rng.random(n)
    y = np.empty(n)
    t_ser = bench(serial_op, x, y)
    y_ser = y.copy()
    t_par = bench(par_op, x, y)
    assert np.allclose(y_ser, y), "parallel result differs from serial"
    return t_ser, t_par


def fmt_time(s):
    if s < 1e-3:
        return f"{s * 1e6:.2f} us"
    return f"{s * 1e3:.3f} ms"


def find_crossover(points):
    """points: list of (n, t_ser, t_par) sorted by n.
    Returns (last_slower_n, first_faster_n) where parallel stays faster from
    first_faster_n onward, or a status string."""
    faster = [t_par < t_ser for _, t_ser, t_par in points]
    if all(faster):
        return "always_faster"
    if not any(faster):
        return "never_faster"
    # first index from which every following size is faster
    idx = None
    for i in range(len(points) - 1, -1, -1):
        if faster[i]:
            idx = i
        else:
            break
    if idx is None:
        return "never_faster"
    if idx == 0:
        return "always_faster"
    return points[idx - 1][0], points[idx][0]


def main():
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Operation     : y[i] = {A} * x[i] + {B}")
    print(f"Parallel      : ThreadPoolExecutor, {WORKERS} workers (pool created once, reused)")
    print(f"Timing        : median of {BATCHES} batches, adaptive repeats\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        par_op = make_parallel(pool, WORKERS)

        points = []
        print(f"{'N':>12} | {'Serial':>12} | {'Parallel':>12} | {'Speedup':>8} | Winner")
        print("-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+" + "-" * 9)
        for n in SIZES:
            t_ser, t_par = measure(n, par_op)
            points.append((n, t_ser, t_par))
            win = "parallel" if t_par < t_ser else "serial"
            print(f"{n:>12,} | {fmt_time(t_ser):>12} | {fmt_time(t_par):>12} | "
                  f"{t_ser / t_par:>8.2f} | {win}")

        print()
        res = find_crossover(points)
        if res == "never_faster":
            print("RESULT: multi-threading was never faster than serial for any tested N.")
            print(f"        (largest N tested: {SIZES[-1]:,})")
            return
        if res == "always_faster":
            print("RESULT: multi-threading was faster than serial for every tested N "
                  f"(from {SIZES[0]:,}).")
            return

        lo, hi = res
        print(f"RESULT: crossover lies between N = {lo:,} (parallel slower) "
              f"and N = {hi:,} (parallel faster).")

        # Refine inside the bracket with a log-spaced scan
        grid = sorted({int(round(v)) for v in np.geomspace(lo, hi, 14)})
        refined = []
        print(f"\nRefinement scan inside ({lo:,} .. {hi:,}):")
        print(f"{'N':>12} | {'Serial':>12} | {'Parallel':>12} | {'Speedup':>8} | Winner")
        print("-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+" + "-" * 9)
        for n in grid:
            t_ser, t_par = measure(n, par_op)
            refined.append((n, t_ser, t_par))
            win = "parallel" if t_par < t_ser else "serial"
            print(f"{n:>12,} | {fmt_time(t_ser):>12} | {fmt_time(t_par):>12} | "
                  f"{t_ser / t_par:>8.2f} | {win}")

        res2 = find_crossover(refined)
        print()
        if isinstance(res2, tuple):
            print(f"REFINED RESULT: tipping point is between N = {res2[0]:,} and N = {res2[1]:,} "
                  f"(parallel faster from ~{res2[1]:,} upward).")
        elif res2 == "always_faster":
            print(f"REFINED RESULT: parallel already faster at N = {lo:,}+ in the fine scan; "
                  f"tipping point is at or just above N = {lo:,}.")
        else:
            print(f"REFINED RESULT: parallel never won inside the fine scan; "
                  f"tipping point is at or just below N = {hi:,}.")
        print("Note: near the tipping point the two timings are close, so the exact value "
              "varies a little between runs.")


if __name__ == "__main__":
    main()
