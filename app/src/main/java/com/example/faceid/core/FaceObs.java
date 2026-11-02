package com.example.faceid.core;

/** One detected face at one sampled frame. Box coordinates are normalized to [0..1]. */
public final class FaceObs {
    public final long tMs;
    public final float x0, y0, x1, y1;
    /** L2-normalized identity embedding, or null if the face was too small / poor quality. */
    public final float[] emb;
    /** 0..1, higher = larger, more frontal face. */
    public final float quality;

    public FaceObs(long tMs, float x0, float y0, float x1, float y1, float[] emb, float quality) {
        this.tMs = tMs;
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
        this.emb = emb;
        this.quality = quality;
    }

    public float width() { return Math.max(0f, x1 - x0); }
    public float height() { return Math.max(0f, y1 - y0); }
    public float area() { return width() * height(); }
    public float cx() { return (x0 + x1) * 0.5f; }
    public float cy() { return (y0 + y1) * 0.5f; }

    public static float iou(FaceObs a, FaceObs b) {
        float ix = Math.max(0f, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
        float iy = Math.max(0f, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
        float inter = ix * iy;
        float union = a.area() + b.area() - inter;
        return union <= 0f ? 0f : inter / union;
    }
}
