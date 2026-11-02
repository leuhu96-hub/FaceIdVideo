package com.example.faceid.core;

/**
 * Least-squares 2D similarity transform (Umeyama, no reflection) mapping src points to dst points.
 * Result {a, b, tx, ty}:  x' = a*x - b*y + tx,  y' = b*x + a*y + ty.
 */
public final class SimilarityTransform {
    private SimilarityTransform() {}

    /** @param src,dst interleaved x,y coordinates, same number of points (>= 2). */
    public static float[] estimate(float[] src, float[] dst) {
        int n = src.length / 2;
        double pmx = 0, pmy = 0, qmx = 0, qmy = 0;
        for (int i = 0; i < n; i++) {
            pmx += src[2 * i];
            pmy += src[2 * i + 1];
            qmx += dst[2 * i];
            qmy += dst[2 * i + 1];
        }
        pmx /= n;
        pmy /= n;
        qmx /= n;
        qmy /= n;
        double num1 = 0, num2 = 0, den = 0;
        for (int i = 0; i < n; i++) {
            double px = src[2 * i] - pmx, py = src[2 * i + 1] - pmy;
            double qx = dst[2 * i] - qmx, qy = dst[2 * i + 1] - qmy;
            num1 += px * qx + py * qy;
            num2 += px * qy - py * qx;
            den += px * px + py * py;
        }
        if (den < 1e-9) return new float[]{1f, 0f, (float) (qmx - pmx), (float) (qmy - pmy)};
        double a = num1 / den, b = num2 / den;
        double tx = qmx - (a * pmx - b * pmy);
        double ty = qmy - (b * pmx + a * pmy);
        return new float[]{(float) a, (float) b, (float) tx, (float) ty};
    }
}
