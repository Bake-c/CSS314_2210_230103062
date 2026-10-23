"""Task 3 - Prefix scan: serial vs. 2-pass chunked parallel scan.

2,000,000 int64 values in [1, 10] from a Student-ID seed.
Serial   : inclusive prefix sum, out[i] = out[i-1] + in[i], one thread (np.cumsum).
Parallel : pass 1 - each worker scans its own chunk locally (and its last value is the
           chunk total); then an exclusive scan of the chunk totals gives each chunk's
           offset; pass 2 - each worker adds its offset to its chunk.
Exit code 1 if parallel_output[-1] != serial_output[-1].

Usage: python task3p.py [workers]    (default: logical core count)
"""
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103062
N = 2_000_000
REPEATS = 7


def serial_scan(a):
    return np.cumsum(a)


def pass1(a, out, lo, hi):
    np.cumsum(a[lo:hi], out=out[lo:hi])      # local inclusive scan of the chunk


def pass2(out, lo, hi, offset):
    if offset:
        out[lo:hi] += offset                 # apply the chunk's offset


def parallel_scan(a, out, pool, workers):
    bounds = np.linspace(0, len(a), workers + 1, dtype=np.int64)
    ranges = [(int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]

    # Pass 1: local scans in parallel
    for f in [pool.submit(pass1, a, out, lo, hi) for lo, hi in ranges]:
        f.result()

    # Offset reduction (tiny serial step over `workers` chunk totals)
    offsets = [0] * workers
    running = 0
    for i, (lo, hi) in enumerate(ranges):
        offsets[i] = running
        running += int(out[hi - 1])

    # Pass 2: add offsets in parallel
    for f in [pool.submit(pass2, out, lo, hi, offsets[i]) for i, (lo, hi) in enumerate(ranges)]:
        f.result()
    return out


def median_ms(fn):
    fn()  # warm-up
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else (os.cpu_count() or 1)
    print(f"Student ID    : {STUDENT_ID}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Workers       : {workers}")
    print(f"Array size    : {N:,} integers in [1, 10]")
    print(f"Timing        : median of {REPEATS} runs after warm-up\n")

    data = np.random.default_rng(STUDENT_ID).integers(1, 11, size=N, dtype=np.int64)
    par_out = np.empty_like(data)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        t_serial = median_ms(lambda: serial_scan(data))
        t_parallel = median_ms(lambda: parallel_scan(data, par_out, pool, workers))
        serial_out = serial_scan(data)
        parallel_out = parallel_scan(data, par_out, pool, workers)

    ok_last = int(parallel_out[-1]) == int(serial_out[-1])
    ok_all = bool(np.array_equal(parallel_out, serial_out))
    status = "PASS" if (ok_last and ok_all) else "FAIL"

    print(f"Last element  : serial = {int(serial_out[-1]):,} | parallel = {int(parallel_out[-1]):,}")
    print(f"Full array equal: {ok_all}\n")
    print(f"{'Serial Time (ms)':>17} | {'Parallel Time (ms)':>19} | {'Speedup':>8} | Verification Status")
    print("-" * 18 + "+" + "-" * 21 + "+" + "-" * 10 + "+" + "-" * 21)
    print(f"{t_serial:>17.3f} | {t_parallel:>19.3f} | {t_serial / t_parallel:>8.2f} | {status}")

    if status != "PASS":
        print("ERROR: parallel scan result does not match serial result", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
