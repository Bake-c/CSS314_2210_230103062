"""Task 4 - Work stealing & scheduling under skew.

1,000 items, cost(i) = sleep(0.0001 * (i % 20)), 4 workers.
  A: static equal blocks
  B: dynamic shared queue of chunks (chunk = 1 and chunk = 5)
time.sleep releases the GIL, so threads really overlap.
"""
import os
import queue
import statistics
import threading
import time

N_ITEMS = 1000
WORKERS = 4
REPEATS = 3
UNIT = 0.0001


def cost(i):
    return UNIT * (i % 20)


def process(i):
    time.sleep(cost(i))


def run_static():
    block = N_ITEMS // WORKERS
    ranges = [(w * block, N_ITEMS if w == WORKERS - 1 else (w + 1) * block)
              for w in range(WORKERS)]
    finish = [0.0] * WORKERS
    counts = [0] * WORKERS

    def worker(w):
        lo, hi = ranges[w]
        for i in range(lo, hi):
            process(i)
        counts[w] = hi - lo
        finish[w] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    return total, finish, counts


def run_dynamic(chunk):
    q = queue.Queue()
    for lo in range(0, N_ITEMS, chunk):
        q.put((lo, min(lo + chunk, N_ITEMS)))
    finish = [0.0] * WORKERS
    counts = [0] * WORKERS

    def worker(w):
        done = 0
        while True:
            try:
                lo, hi = q.get_nowait()
            except queue.Empty:
                break
            for i in range(lo, hi):
                process(i)
            done += hi - lo
        counts[w] = done
        finish[w] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    return total, finish, counts


def median_run(fn, *args):
    runs = sorted((fn(*args) for _ in range(REPEATS)), key=lambda r: r[0])
    return runs[len(runs) // 2]


def report(name, total, finish, counts):
    fmin, fmax = min(finish), max(finish)
    print(f"{name}")
    print(f"  Total time        : {total * 1000:9.2f} ms")
    print(f"  First thread done : {fmin * 1000:9.2f} ms")
    print(f"  Last thread done  : {fmax * 1000:9.2f} ms")
    print(f"  Delta (max - min) : {(fmax - fmin) * 1000:9.2f} ms")
    print("  Per-worker finish : " + ", ".join(f"{f * 1000:.1f}" for f in finish) + " ms")
    print("  Items per worker  : " + ", ".join(str(c) for c in counts))
    print()


def main():
    total_work = sum(cost(i) for i in range(N_ITEMS))
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Items         : {N_ITEMS}, cost(i) = sleep(0.0001 * (i % 20)), workers = {WORKERS}")
    print(f"Total work    : {total_work * 1000:.2f} ms  (serial ideal)")
    print(f"Perfect split : {total_work / WORKERS * 1000:.2f} ms  (work / {WORKERS})")
    print(f"Timing        : median run of {REPEATS} per strategy\n")

    process(1)  # warm-up
    results = [
        ("Strategy A: static equal blocks (250 items each)", median_run(run_static)),
        ("Strategy B1: dynamic queue, chunk = 1", median_run(run_dynamic, 1)),
        ("Strategy B2: dynamic queue, chunk = 5", median_run(run_dynamic, 5)),
    ]
    for name, (total, finish, counts) in results:
        report(name, total, finish, counts)

    print("Summary")
    print(f"{'Strategy':<22} | {'Total (ms)':>10} | {'Delta (ms)':>10}")
    print("-" * 23 + "+" + "-" * 12 + "+" + "-" * 11)
    for label, (name, (total, finish, _)) in zip(["A static", "B1 dynamic c=1", "B2 dynamic c=5"], results):
        print(f"{label:<22} | {total * 1000:>10.2f} | {(max(finish) - min(finish)) * 1000:>10.2f}")


if __name__ == "__main__":
    main()
