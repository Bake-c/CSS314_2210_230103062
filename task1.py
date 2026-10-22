"""Task 1 - Hardware probe & serial baseline."""
import os
import platform
import time

import numpy as np

STUDENT_ID = 230103062
N = 20_000_000


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()} ({platform.platform()})")
    print(f"CPU model        : {cpu_model()}")
    print(f"Logical cores    : {os.cpu_count()}")
    print(f"Array size       : {N:,} float64 elements")

    t0 = time.perf_counter()
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)
    gen_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Generation time  : {gen_ms:.2f} ms (not included in compute time)")

    # Serial compute: single thread, sum + standard deviation
    t0 = time.perf_counter()
    total = float(np.sum(data))
    mean = total / N
    var = float(np.sum((data - mean) ** 2)) / N  # population std
    std = var ** 0.5
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Sum              : {total:.6f}")
    print(f"Std deviation    : {std:.9f}")
    print(f"Serial wall time : {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
