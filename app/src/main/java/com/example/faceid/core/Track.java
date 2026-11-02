package com.example.faceid.core;

import java.util.ArrayList;
import java.util.List;

/** A face track: consecutive observations of (presumably) the same face inside one shot. */
public final class Track {
    public final int id;
    public final List<FaceObs> obs = new ArrayList<>();
    private float[] embSum;
    private float embWeight;
    private float[] lastEmb;

    public Track(int id) {
        this.id = id;
    }

    public void add(FaceObs o) {
        obs.add(o);
        if (o.emb != null) {
            if (embSum == null) embSum = new float[o.emb.length];
            float w = Math.max(1e-3f, o.quality);
            for (int i = 0; i < embSum.length; i++) embSum[i] += w * o.emb[i];
            embWeight += w;
            lastEmb = o.emb;
        }
    }

    public FaceObs last() { return obs.get(obs.size() - 1); }
    public long startMs() { return obs.get(0).tMs; }
    public long endMs() { return last().tMs; }
    public int size() { return obs.size(); }
    public boolean hasEmbedding() { return embSum != null; }
    public float[] lastEmbedding() { return lastEmb; }

    /** Quality-weighted mean embedding, L2-normalized (copy), or null. */
    public float[] embedding() {
        if (embSum == null) return null;
        return Vec.normalize(embSum.clone());
    }
}
