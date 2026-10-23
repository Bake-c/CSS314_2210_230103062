import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Task 2 - Pseudo-random generator state bottleneck (Monte Carlo pi).
 * 40,000,000 points, 4 threads.
 *   Shared     : ONE SplittableRandom (not thread-safe) guarded by a ReentrantLock.
 *   Independent: one SplittableRandom per thread, seed = (StudentID * 10) + threadIndex.
 */
public class Task2p {
    static final long STUDENT_ID = 230103062L;
    static final long TOTAL_POINTS = 40_000_000L;
    static final int THREADS = 4;
    static final int RUNS = 3;

    interface Worker { long run(int threadIndex, long points); }

    static final class Result {
        final double pi; final double ms;
        Result(double pi, double ms) { this.pi = pi; this.ms = ms; }
    }

    static Result execute(Worker worker, long totalPoints) throws InterruptedException {
        long per = totalPoints / THREADS;
        long[] hits = new long[THREADS];
        Thread[] ts = new Thread[THREADS];
        CountDownLatch ready = new CountDownLatch(THREADS);
        CountDownLatch go = new CountDownLatch(1);
        for (int i = 0; i < THREADS; i++) {
            final int id = i;
            ts[i] = new Thread(() -> {
                ready.countDown();
                try { go.await(); } catch (InterruptedException e) { return; }
                hits[id] = worker.run(id, per);
            });
            ts[i].start();
        }
        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread t : ts) t.join();
        double ms = (System.nanoTime() - t0) / 1e6;
        long total = 0;
        for (long h : hits) total += h;
        return new Result(4.0 * total / (per * THREADS), ms);
    }

    static Result sharedRun(long totalPoints) throws InterruptedException {
        final SplittableRandom shared = new SplittableRandom(STUDENT_ID * 10);
        final ReentrantLock lock = new ReentrantLock();
        return execute((id, points) -> {
            long inside = 0;
            for (long k = 0; k < points; k++) {
                double x, y;
                lock.lock();               // every sample contends for the generator state
                try {
                    x = shared.nextDouble();
                    y = shared.nextDouble();
                } finally {
                    lock.unlock();
                }
                if (x * x + y * y <= 1.0) inside++;
            }
            return inside;
        }, totalPoints);
    }

    static Result independentRun(long totalPoints) throws InterruptedException {
        return execute((id, points) -> {
            SplittableRandom rng = new SplittableRandom(STUDENT_ID * 10 + id);
            long inside = 0;
            for (long k = 0; k < points; k++) {
                double x = rng.nextDouble();
                double y = rng.nextDouble();
                if (x * x + y * y <= 1.0) inside++;
            }
            return inside;
        }, totalPoints);
    }

    static Result median(Result[] r) {
        Result[] c = r.clone();
        Arrays.sort(c, (a, b) -> Double.compare(a.ms, b.ms));
        return c[c.length / 2];
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Student ID     : " + STUDENT_ID);
        System.out.println("Logical cores  : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads        : " + THREADS);
        System.out.println("Sample points  : " + String.format("%,d", TOTAL_POINTS));
        System.out.println("Seeds (indep.) : (" + STUDENT_ID + " * 10) + threadIndex = "
                + (STUDENT_ID * 10) + " .. " + (STUDENT_ID * 10 + THREADS - 1));
        System.out.println("Timing         : median of " + RUNS + " runs after warm-up\n");

        // warm-up (JIT)
        sharedRun(2_000_000L);
        independentRun(2_000_000L);

        Result[] sh = new Result[RUNS], in = new Result[RUNS];
        for (int r = 0; r < RUNS; r++) {
            sh[r] = sharedRun(TOTAL_POINTS);
            in[r] = independentRun(TOTAL_POINTS);
        }
        Result s = median(sh), i = median(in);

        System.out.println("Shared PRNG (one generator + lock):");
        System.out.printf("  pi = %.8f | |error| = %.8f | wall time = %.2f ms%n",
                s.pi, Math.abs(s.pi - Math.PI), s.ms);
        System.out.println("Independent PRNG per thread:");
        System.out.printf("  pi = %.8f | |error| = %.8f | wall time = %.2f ms%n",
                i.pi, Math.abs(i.pi - Math.PI), i.ms);
        System.out.printf("%nSpeedup from removing generator locking: %.2fx%n", s.ms / i.ms);
    }
}
