"""Task 7 - Arithmetic intensity vs. memory bottlenecks.

Kernel A (memory-bound):  B[i] = A[i]                         (vector copy)
Kernel B (compute-bound): B[i] = sin(A[i])*cos(A[i]) + tan(A[i])
N = 30,000,000 float64 elements; worker counts 1, 2 and max logical cores.
numpy releases the GIL inside its loops, so threads run in parallel.
"""
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N = 30_000_000
REPEATS = 5
SUB = 1 << 18  # 262,144-element sub-blocks keep kernel B temporaries cache-sized


def kernel_a(a, b, lo, hi):
    np.copyto(b[lo:hi], a[lo:hi])


def kernel_b(a, b, lo, hi):
    for s in range(lo, hi, SUB):
        e = min(s + SUB, hi)
        x = a[s:e]
        sn = np.sin(x)
        cs = np.cos(x)
        tn = np.tan(x)
        np.multiply(sn, cs, out=sn)
        np.add(sn, tn, out=b[s:e])


def run(kernel, a, b, workers):
    bounds = np.linspace(0, len(a), workers + 1, dtype=np.int64)
    if workers == 1:
        kernel(a, b, 0, len(a))
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(kernel, a, b, int(bounds[i]), int(bounds[i + 1]))
                   for i in range(workers)]
        for f in futures:
            f.result()


def bench(kernel, a, b, workers):
    run(kernel, a, b, workers)  # warm-up
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        run(kernel, a, b, workers)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    max_cores = os.cpu_count() or 1
    counts = sorted({1, 2, max_cores})

    print(f"Logical cores : {max_cores}")
    print(f"Array size    : {N:,} float64 ({N * 8 / 1e6:.0f} MB per array)")
    print(f"Worker counts : {counts}")
    print(f"Timing        : median of {REPEATS} runs\n")

    rng = np.random.default_rng(1)
    a = rng.random(N)
    b = np.empty(N)

    kernels = [
        ("Kernel A (memory-bound): B[i] = A[i]", kernel_a),
        ("Kernel B (compute-bound): sin*cos+tan", kernel_b),
    ]
    eff_at_max = {}
    for name, fn in kernels:
        print(name)
        print(f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Efficiency':>10}")
        print("-" * 7 + "+" + "-" * 16 + "+" + "-" * 10 + "+" + "-" * 11)
        t1 = None
        for p in counts:
            t = bench(fn, a, b, p)
            if t1 is None:
                t1 = t
            speedup = t1 / t
            eff = speedup / p
            print(f"{p:>6} | {t:>14.2f} | {speedup:>8.2f} | {eff:>10.2f}")
            if p == counts[-1]:
                eff_at_max[name] = eff
        print()

    ea = eff_at_max[kernels[0][0]]
    eb = eff_at_max[kernels[1][0]]
    p = counts[-1]
    print("CONCLUSION")
    print(f"  Efficiency at {p} cores: Kernel A (copy) = {ea:.2f}, Kernel B (sin*cos+tan) = {eb:.2f}")
    if p == 1:
        print("  Only 1 core available: scaling cannot be assessed.")
    elif ea < eb and ea < 0.75:
        print("  Kernel A (memory-bound vector copy) STALLED in scaling: its threads")
        print("  saturate the memory channel / RAM bandwidth, so extra cores add little.")
        print("  Kernel B (compute-bound) keeps scaling because its time is spent in the CPU.")
    elif ea < eb:
        print("  Kernel A (memory-bound vector copy) scales worse than Kernel B, so it is the")
        print("  kernel limited by the memory channel / RAM bandwidth; here the saturation is")
        print("  only partial (efficiency still >= 0.75). It gets worse as more cores share the bus.")
        print("  Kernel B (compute-bound) scales almost linearly because its time is spent in the CPU.")
    else:
        print("  Kernel B scaled worse than Kernel A on this run; no clear memory-channel")
        print("  saturation was observed for the copy kernel (try re-running on an idle machine).")


if __name__ == "__main__":
    main()
