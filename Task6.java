import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Task 6 - Synchronization cost & data loss.
 * 4 threads x 2,000,000 increments of one shared counter.
 *   1) no synchronization (volatile int, counter++ is read-modify-write -> races)
 *   2) AtomicInteger (hardware atomic primitive)
 *   3) ReentrantLock (mutex)
 */
public class Task6 {
    static final int THREADS = 4;
    static final int INCREMENTS = 2_000_000;
    static final long EXPECTED = (long) THREADS * INCREMENTS;
    static final int RUNS = 3;

    // volatile forces a real load/store each iteration, so the JIT cannot
    // collapse the loop into a register; counter++ is still NOT atomic.
    static volatile int unsafeCounter;
    static final AtomicInteger atomicCounter = new AtomicInteger();
    static int lockedCounter;
    static final ReentrantLock lock = new ReentrantLock();

    static long runThreads(Runnable body) throws InterruptedException {
        Thread[] ts = new Thread[THREADS];
        CountDownLatch ready = new CountDownLatch(THREADS);
        CountDownLatch go = new CountDownLatch(1);
        for (int i = 0; i < THREADS; i++) {
            ts[i] = new Thread(() -> {
                ready.countDown();
                try { go.await(); } catch (InterruptedException e) { return; }
                body.run();
            });
            ts[i].start();
        }
        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread t : ts) t.join();
        return System.nanoTime() - t0;
    }

    static long unsafeRun() throws InterruptedException {
        unsafeCounter = 0;
        return runThreads(() -> {
            for (int k = 0; k < INCREMENTS; k++) unsafeCounter++;
        });
    }

    static long atomicRun() throws InterruptedException {
        atomicCounter.set(0);
        return runThreads(() -> {
            for (int k = 0; k < INCREMENTS; k++) atomicCounter.incrementAndGet();
        });
    }

    static long lockRun() throws InterruptedException {
        lockedCounter = 0;
        return runThreads(() -> {
            for (int k = 0; k < INCREMENTS; k++) {
                lock.lock();
                try { lockedCounter++; } finally { lock.unlock(); }
            }
        });
    }

    static double median(long[] a) {
        long[] c = a.clone();
        Arrays.sort(c);
        return c[c.length / 2] / 1e6;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Logical cores : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads       : " + THREADS);
        System.out.println("Increments    : " + String.format("%,d", INCREMENTS) + " per thread");
        System.out.println("Expected final: " + String.format("%,d", EXPECTED));
        System.out.println();

        // warm-up (JIT)
        unsafeRun(); atomicRun(); lockRun();

        long[] tUnsafe = new long[RUNS], tAtomic = new long[RUNS], tLock = new long[RUNS];

        System.out.println("=== 1) No synchronization (race condition) ===");
        for (int r = 0; r < RUNS; r++) {
            tUnsafe[r] = unsafeRun();
            long actual = unsafeCounter;
            long lost = EXPECTED - actual;
            System.out.printf("Run %d: expected %,d | actual %,d | lost %,d updates (%.2f%%) | %.2f ms%n",
                    r + 1, EXPECTED, actual, lost, 100.0 * lost / EXPECTED, tUnsafe[r] / 1e6);
        }

        System.out.println("\n=== 2) AtomicInteger (hardware atomic) ===");
        for (int r = 0; r < RUNS; r++) {
            tAtomic[r] = atomicRun();
            long actual = atomicCounter.get();
            System.out.printf("Run %d: expected %,d | actual %,d | lost %,d (%.2f%%) | %.2f ms%n",
                    r + 1, EXPECTED, actual, EXPECTED - actual,
                    100.0 * (EXPECTED - actual) / EXPECTED, tAtomic[r] / 1e6);
        }

        System.out.println("\n=== 3) ReentrantLock (mutex) ===");
        for (int r = 0; r < RUNS; r++) {
            tLock[r] = lockRun();
            long actual = lockedCounter;
            System.out.printf("Run %d: expected %,d | actual %,d | lost %,d (%.2f%%) | %.2f ms%n",
                    r + 1, EXPECTED, actual, EXPECTED - actual,
                    100.0 * (EXPECTED - actual) / EXPECTED, tLock[r] / 1e6);
        }

        double mu = median(tUnsafe), ma = median(tAtomic), ml = median(tLock);
        System.out.println("\n=== Synchronization penalty (median wall time) ===");
        System.out.printf("%-22s | %10s | %12s | %8s%n", "Version", "Time (ms)", "Penalty (ms)", "Slowdown");
        System.out.println("-----------------------+------------+--------------+---------");
        System.out.printf("%-22s | %10.2f | %12s | %8s%n", "No sync (incorrect)", mu, "-", "1.00x");
        System.out.printf("%-22s | %10.2f | %12.2f | %7.2fx%n", "AtomicInteger", ma, ma - mu, ma / mu);
        System.out.printf("%-22s | %10.2f | %12.2f | %7.2fx%n", "ReentrantLock", ml, ml - mu, ml / mu);
    }
}
