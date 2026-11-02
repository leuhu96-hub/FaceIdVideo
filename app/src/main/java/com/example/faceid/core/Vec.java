package com.example.faceid.core;

public final class Vec {
    private Vec() {}

    public static float dot(float[] a, float[] b) {
        float s = 0f;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    /** Normalizes in place and returns the same array. */
    public static float[] normalize(float[] v) {
        double s = 0;
        for (float x : v) s += x * x;
        float n = (float) Math.sqrt(s);
        if (n > 1e-12f) {
            for (int i = 0; i < v.length; i++) v[i] /= n;
        }
        return v;
    }
}
