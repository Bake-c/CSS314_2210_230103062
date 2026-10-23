"""Task 4 (CPU-only path) - IPC overhead: pickled pipe vs. zero-copy shared memory.

A persistent worker process is started once (start-up cost is not measured).
For each N (float64 array) the parent hands the array to the worker and waits for
the worker's acknowledgement (it reads the first and last element to prove it saw the data).

  Path A   : Pipe + pickle serialization (the array's bytes are copied through the pipe).
  Path B   : multiprocessing.shared_memory. Timed: create segment + copy array into it +
             worker attaches + ack. (The one copy is needed to get ordinary data into the segment.)
  Path B0  : same, but the data already lives in the shared segment (producer wrote it there
             in place), so only "send name + worker attaches + ack" is timed: true zero-copy.
"""
import multiprocessing as mp
import os
import platform
import statistics
import time
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103062
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 7


def attach(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # Python 3.13+
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def worker(conn):
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "A":
            arr = msg[1]
            conn.send(float(arr[0]) + float(arr[-1]))
        elif kind == "B":
            name, n = msg[1], msg[2]
            shm = attach(name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            result = float(arr[0]) + float(arr[-1])
            del arr
            shm.close()
            conn.send(result)
        elif kind == "STOP":
            return


def time_path_a(conn, data):
    t0 = time.perf_counter()
    conn.send(("A", data))
    ack = conn.recv()
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, ack


def time_path_b(conn, data):
    t0 = time.perf_counter()
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    view = np.ndarray(data.shape, dtype=np.float64, buffer=shm.buf)
    view[:] = data
    conn.send(("B", shm.name, len(data)))
    ack = conn.recv()
    ms = (time.perf_counter() - t0) * 1000.0
    del view
    shm.close()
    shm.unlink()
    return ms, ack


def time_path_b0(conn, shm, n):
    t0 = time.perf_counter()
    conn.send(("B", shm.name, n))
    ack = conn.recv()
    return (time.perf_counter() - t0) * 1000.0, ack


def median_of(fn):
    fn()  # warm-up
    results = [fn() for _ in range(REPEATS)]
    return statistics.median(r[0] for r in results), results[-1][1]


def main():
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"Logical cores : {os.cpu_count()}")
    print("Mode          : CPU only (inter-process, no GPU)")
    print(f"Timing        : median of {REPEATS} runs after warm-up; worker started once\n")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=worker, args=(child_conn,), daemon=True)
    proc.start()
    child_conn.close()

    rng = np.random.default_rng(STUDENT_ID)
    rows = []
    try:
        for n in SIZES:
            data = rng.random(n)
            expected = float(data[0]) + float(data[-1])

            shm0 = shared_memory.SharedMemory(create=True, size=data.nbytes)
            np.ndarray(data.shape, dtype=np.float64, buffer=shm0.buf)[:] = data

            ms_a, ack_a = median_of(lambda: time_path_a(parent_conn, data))
            ms_b, ack_b = median_of(lambda: time_path_b(parent_conn, data))
            ms_b0, ack_b0 = median_of(lambda: time_path_b0(parent_conn, shm0, n))

            assert ack_a == ack_b == ack_b0 == expected, "worker saw different data"
            shm0.close()
            shm0.unlink()
            rows.append((n, data.nbytes / 1e6, ms_a, ms_b, ms_b0))
    finally:
        parent_conn.send(("STOP",))
        proc.join(timeout=10)

    print("Transfer overhead (ms), lower is better")
    print(f"{'N':>12} | {'Size (MB)':>9} | {'A: Pipe+pickle':>14} | {'B: shm (+copy)':>14} | "
          f"{'B0: shm zero-copy':>17} | {'A / B':>7} | {'A / B0':>8}")
    print("-" * 13 + "+" + "-" * 11 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 19 + "+" + "-" * 9 + "+" + "-" * 9)
    for n, mb, a, b, b0 in rows:
        print(f"{n:>12,} | {mb:>9.1f} | {a:>14.3f} | {b:>14.3f} | {b0:>17.3f} | {a / b:>6.1f}x | {a / b0:>7.1f}x")

    print("\nConclusion: serializing through the pipe costs time proportional to the array size "
          "(bytes are copied twice, user -> pipe -> worker),")
    print("while shared memory pays at most one copy, and zero-copy shared memory stays almost "
          "constant (only the name is sent).")


if __name__ == "__main__":
    main()
