"""Task 2 - Multi-core scaling vs. Amdahl's Law (P = 0.95)."""
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103062
N = 20_000_000
P = 0.95
REPEATS = 5


def chunk_sum(arr, lo, hi):
    # np.sum releases the GIL, so threads run truly in parallel
    return float(np.sum(arr[lo:hi]))


def parallel_sum(arr, workers):
    bounds = np.linspace(0, len(arr), workers + 1, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(chunk_sum, arr, int(bounds[i]), int(bounds[i + 1]))
                   for i in range(workers)]
        return sum(f.result() for f in futures)


def bench(arr, workers):
    times = []
    result = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = parallel_sum(arr, workers)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), result


def amdahl(p_workers):
    return 1.0 / ((1.0 - P) + P / p_workers)


def main():
    max_cores = os.cpu_count() or 1
    counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    print(f"Student ID    : {STUDENT_ID}")
    print(f"Logical cores : {max_cores}")
    print(f"Array size    : {N:,} float64, parallel fraction P = {P}")
    print(f"Timing        : median of {REPEATS} runs per worker count\n")

    arr = np.random.default_rng(STUDENT_ID).random(N, dtype=np.float64)
    parallel_sum(arr, 1)  # warm-up

    results = {}
    for w in counts:
        results[w] = bench(arr, w)
    t1 = results[1][0]

    print(f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}")
    print("-" * 7 + "+" + "-" * 16 + "+" + "-" * 10 + "+" + "-" * 20)
    for w in counts:
        t, _ = results[w]
        print(f"{w:>6} | {t:>14.2f} | {t1 / t:>8.2f} | {amdahl(w):>18.2f}")

    print(f"\nChecksum (sum, 1 worker): {results[1][1]:.6f}")
    print(f"Amdahl limit as p -> inf : {1.0 / (1.0 - P):.1f}x")


if __name__ == "__main__":
    main()
