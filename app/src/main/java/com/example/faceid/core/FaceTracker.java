package com.example.faceid.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/**
 * Lightweight IoU + appearance tracker (ByteTrack-like greedy association, without a Kalman filter).
 * Tracks are always cut at scene cuts; re-identification across cuts is the clusterer's job.
 */
public final class FaceTracker {

    public static final class Params {
        /** Minimum IoU between the track's last box and a new detection. */
        public float iouThreshold = 0.25f;
        /** A track that is not seen for longer than this is closed. */
        public long maxGapMs = 700;
        /** If both have embeddings, refuse IoU matches whose cosine similarity is below this. */
        public float embGate = 0.20f;
        /** Second pass: match by appearance when IoU fails (fast motion), if similarity >= this ... */
        public float embMatch = 0.55f;
        /** ... and the centers are closer than this many face sizes. */
        public float maxCenterDistInFaces = 1.5f;
    }

    private final Params p;
    private final List<Track> active = new ArrayList<>();
    private final List<Track> finished = new ArrayList<>();
    private int nextId = 0;

    public FaceTracker(Params params) {
        this.p = params;
    }

    /** @return the track id assigned to each detection, same order as {@code dets}. */
    public int[] update(long tMs, List<FaceObs> dets, boolean sceneCut) {
        if (sceneCut) {
            finished.addAll(active);
            active.clear();
        }
        for (Iterator<Track> it = active.iterator(); it.hasNext(); ) {
            Track t = it.next();
            if (tMs - t.endMs() > p.maxGapMs) {
                finished.add(t);
                it.remove();
            }
        }

        int[] assign = new int[dets.size()];
        Arrays.fill(assign, -1);
        boolean[] tUsed = new boolean[active.size()];
        boolean[] dUsed = new boolean[dets.size()];

        // Pass 1: IoU, gated by appearance.
        List<float[]> cand = new ArrayList<>();
        for (int ti = 0; ti < active.size(); ti++) {
            Track t = active.get(ti);
            FaceObs last = t.last();
            float[] le = t.lastEmbedding();
            for (int di = 0; di < dets.size(); di++) {
                FaceObs d = dets.get(di);
                float iou = FaceObs.iou(last, d);
                if (iou < p.iouThreshold) continue;
                if (le != null && d.emb != null && Vec.dot(le, d.emb) < p.embGate) continue;
                cand.add(new float[]{iou, ti, di});
            }
        }
        greedy(cand, tUsed, dUsed, dets, assign);

        // Pass 2: appearance + proximity for detections that moved too fast for IoU.
        cand.clear();
        for (int ti = 0; ti < active.size(); ti++) {
            if (tUsed[ti]) continue;
            Track t = active.get(ti);
            float[] le = t.lastEmbedding();
            if (le == null) continue;
            FaceObs last = t.last();
            float faceSize = Math.max(last.width(), last.height());
            for (int di = 0; di < dets.size(); di++) {
                if (dUsed[di]) continue;
                FaceObs d = dets.get(di);
                if (d.emb == null) continue;
                float dx = d.cx() - last.cx(), dy = d.cy() - last.cy();
                float dist = (float) Math.sqrt(dx * dx + dy * dy);
                if (dist > p.maxCenterDistInFaces * faceSize) continue;
                float sim = Vec.dot(le, d.emb);
                if (sim >= p.embMatch) cand.add(new float[]{sim, ti, di});
            }
        }
        greedy(cand, tUsed, dUsed, dets, assign);

        for (int di = 0; di < dets.size(); di++) {
            if (dUsed[di]) continue;
            Track t = new Track(nextId++);
            t.add(dets.get(di));
            active.add(t);
            assign[di] = t.id;
        }
        return assign;
    }

    private void greedy(List<float[]> cand, boolean[] tUsed, boolean[] dUsed,
                        List<FaceObs> dets, int[] assign) {
        cand.sort(new Comparator<float[]>() {
            @Override
            public int compare(float[] a, float[] b) { return Float.compare(b[0], a[0]); }
        });
        for (float[] c : cand) {
            int ti = (int) c[1], di = (int) c[2];
            if (tUsed[ti] || dUsed[di]) continue;
            tUsed[ti] = true;
            dUsed[di] = true;
            Track t = active.get(ti);
            t.add(dets.get(di));
            assign[di] = t.id;
        }
    }

    /** Closes all tracks and returns every track, ordered by id. */
    public List<Track> finish() {
        finished.addAll(active);
        active.clear();
        List<Track> all = new ArrayList<>(finished);
        all.sort(new Comparator<Track>() {
            @Override
            public int compare(Track a, Track b) { return Integer.compare(a.id, b.id); }
        });
        return all;
    }
}
