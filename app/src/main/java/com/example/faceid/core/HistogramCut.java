package com.example.faceid.core;

/** Scene-cut detector: L1 distance between RGB histograms of consecutive sampled frames. */
public final class HistogramCut {
    private static final int BINS = 16;
    private final float threshold;
    private float[] prev;

    /** @param threshold 0..1; ~0.35 works for most phone videos. */
    public HistogramCut(float threshold) {
        this.threshold = threshold;
    }

    /** @param argb pixels of a small thumbnail (e.g. 64x36) of the frame. */
    public boolean update(int[] argb) {
        float[] h = histogram(argb);
        boolean cut = prev != null && distance(prev, h) > threshold;
        prev = h;
        return cut;
    }

    static float[] histogram(int[] argb) {
        float[] h = new float[3 * BINS];
        for (int c : argb) {
            h[((c >> 16) & 0xff) * BINS / 256]++;
            h[BINS + ((c >> 8) & 0xff) * BINS / 256]++;
            h[2 * BINS + (c & 0xff) * BINS / 256]++;
        }
        float n = Math.max(1, argb.length);
        for (int i = 0; i < h.length; i++) h[i] /= n;
        return h;
    }

    /** 0 = identical, 1 = disjoint. */
    static float distance(float[] a, float[] b) {
        float s = 0f;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s / 6f; // 3 channels, each L1 in [0, 2]
    }
}
