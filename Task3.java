import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/**
 * Task 3 - Cache line contention (false sharing).
 * Usage: java Task3 [threads]   (default: number of physical cores)
 */
public class Task3 {
    static final long ITERATIONS = 50_000_000L;
    static final int PAD = 8; // 8 longs * 8 bytes = 64-byte cache line
    static final VarHandle ARR = MethodHandles.arrayElementVarHandle(long[].class);

    static int physicalCores() {
        int logical = Runtime.getRuntime().availableProcessors();
        try {
            Process p = new ProcessBuilder("lscpu", "-p=CORE,SOCKET").redirectErrorStream(true).start();
            java.util.Set<String> uniq = new java.util.HashSet<>();
            try (java.io.BufferedReader r = new java.io.BufferedReader(
                    new java.io.InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) {
                    if (!line.startsWith("#") && !line.isBlank()) uniq.add(line.trim());
                }
            }
            p.waitFor();
            if (!uniq.isEmpty()) return Math.min(uniq.size(), logical);
        } catch (Exception ignored) { }
        return logical;
    }

    interface Body { void run(int i); }

    static long timeThreads(int n, Body body) throws InterruptedException {
        Thread[] ts = new Thread[n];
        for (int i = 0; i < n; i++) {
            final int id = i;
            ts[i] = new Thread(() -> body.run(id));
        }
        long t0 = System.nanoTime();
        for (Thread t : ts) t.start();
        for (Thread t : ts) t.join();
        return System.nanoTime() - t0;
    }

    public static void main(String[] args) throws Exception {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : physicalCores();
        System.out.println("Logical cores  : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads (N)    : " + n + " (physical cores)");
        System.out.println("Iterations     : " + ITERATIONS + " per thread\n");

        long[] shared = new long[n];            // adjacent -> same cache line
        long[] padded = new long[n * PAD];      // 64 bytes between elements
        long[] local = new long[n];

        // Opaque accesses force a real memory load/store on every iteration,
        // so the JIT cannot keep the counter in a register.
        Body unpadded = id -> {
            for (long k = 0; k < ITERATIONS; k++) {
                ARR.setOpaque(shared, id, (long) ARR.getOpaque(shared, id) + 1);
            }
        };
        Body paddedBody = id -> {
            int idx = id * PAD;
            for (long k = 0; k < ITERATIONS; k++) {
                ARR.setOpaque(padded, idx, (long) ARR.getOpaque(padded, idx) + 1);
            }
        };
        Body localBody = id -> {
            long c = 0;
            for (long k = 0; k < ITERATIONS; k++) c++;
            local[id] = c; // single write at the end
        };

        // warm-up (JIT)
        timeThreads(n, unpadded);
        timeThreads(n, paddedBody);
        timeThreads(n, localBody);
        java.util.Arrays.fill(shared, 0);
        java.util.Arrays.fill(padded, 0);

        int reps = 3;
        long bestU = Long.MAX_VALUE, bestP = Long.MAX_VALUE, bestL = Long.MAX_VALUE;
        for (int r = 0; r < reps; r++) {
            bestU = Math.min(bestU, timeThreads(n, unpadded));
            bestP = Math.min(bestP, timeThreads(n, paddedBody));
            bestL = Math.min(bestL, timeThreads(n, localBody));
        }

        double ms = 1e6;
        System.out.printf("Unpadded shared (false sharing) : %10.2f ms%n", bestU / ms);
        System.out.printf("Padded (64-byte stride)         : %10.2f ms%n", bestP / ms);
        System.out.printf("Local variable (1 write at end) : %10.2f ms%n", bestL / ms);
        System.out.printf("%nRatio Unpadded / Padded : %.2fx slower%n", (double) bestU / bestP);
        System.out.printf("Ratio Unpadded / Local  : %.2fx slower%n", (double) bestU / bestL);
        System.out.println("(best of " + reps + " runs after warm-up)");
    }
}
