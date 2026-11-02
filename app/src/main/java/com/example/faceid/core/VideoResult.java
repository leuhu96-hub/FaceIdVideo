package com.example.faceid.core;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/** Final result: tracks + person id per track, queried by playback time. Person ids start at 1. */
public final class VideoResult {

    public static final class Box {
        public final int personId, trackId;
        public final float x0, y0, x1, y1;

        Box(int personId, int trackId, float x0, float y0, float x1, float y1) {
            this.personId = personId;
            this.trackId = trackId;
            this.x0 = x0;
            this.y0 = y0;
            this.x1 = x1;
            this.y1 = y1;
        }

        public boolean contains(float x, float y) { return x >= x0 && x <= x1 && y >= y0 && y <= y1; }
        public float area() { return (x1 - x0) * (y1 - y0); }
    }

    private final List<Track> tracks;
    private final int[] personOfTrack;
    private final long sampleMs;

    public VideoResult(List<Track> tracks, int[] labels, long sampleMs) {
        this.tracks = tracks;
        this.sampleMs = sampleMs;
        this.personOfTrack = new int[labels.length];
        for (int i = 0; i < labels.length; i++) personOfTrack[i] = labels[i] + 1;
    }

    /** Interpolated boxes visible at playback time {@code t}. */
    public List<Box> boxesAt(long t) {
        List<Box> out = new ArrayList<>();
        long tol = sampleMs * 3 / 4;
        long maxInterp = sampleMs * 5 / 2;
        for (int k = 0; k < tracks.size(); k++) {
            Track tr = tracks.get(k);
            if (t < tr.startMs() - tol || t > tr.endMs() + tol) continue;
            List<FaceObs> obs = tr.obs;
            int i = floorIndex(obs, t);
            FaceObs a = i >= 0 ? obs.get(i) : null;
            FaceObs b = i + 1 < obs.size() ? obs.get(i + 1) : null;
            int pid = personOfTrack[k];
            if (a != null && b != null && b.tMs - a.tMs <= maxInterp) {
                float f = (t - a.tMs) / (float) Math.max(1, b.tMs - a.tMs);
                out.add(new Box(pid, tr.id, lerp(a.x0, b.x0, f), lerp(a.y0, b.y0, f),
                        lerp(a.x1, b.x1, f), lerp(a.y1, b.y1, f)));
            } else {
                FaceObs near = null;
                if (a != null && t - a.tMs <= tol) near = a;
                if (b != null && b.tMs - t <= tol && (near == null || b.tMs - t < t - a.tMs)) near = b;
                if (near != null) out.add(new Box(pid, tr.id, near.x0, near.y0, near.x1, near.y1));
            }
        }
        return out;
    }

    private static int floorIndex(List<FaceObs> obs, long t) {
        int lo = 0, hi = obs.size() - 1, ans = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (obs.get(mid).tMs <= t) {
                ans = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return ans;
    }

    private static float lerp(float a, float b, float f) { return a + (b - a) * f; }

    public List<Integer> personIds() {
        TreeSet<Integer> s = new TreeSet<>();
        for (int p : personOfTrack) s.add(p);
        return new ArrayList<>(s);
    }

    /** Track ids (Track.id) belonging to a person. */
    public List<Integer> trackIdsOf(int personId) {
        List<Integer> out = new ArrayList<>();
        for (int k = 0; k < tracks.size(); k++) if (personOfTrack[k] == personId) out.add(tracks.get(k).id);
        return out;
    }

    /** Time intervals [start, end] (ms) where the person is on screen; gaps < 1 s are merged. */
    public List<long[]> segments(int personId) {
        List<long[]> raw = new ArrayList<>();
        for (int k = 0; k < tracks.size(); k++) {
            if (personOfTrack[k] != personId) continue;
            Track tr = tracks.get(k);
            raw.add(new long[]{Math.max(0, tr.startMs() - sampleMs / 2), tr.endMs() + sampleMs / 2});
        }
        raw.sort((a, b) -> Long.compare(a[0], b[0]));
        List<long[]> merged = new ArrayList<>();
        for (long[] s : raw) {
            if (!merged.isEmpty() && s[0] - merged.get(merged.size() - 1)[1] < 1000) {
                long[] last = merged.get(merged.size() - 1);
                last[1] = Math.max(last[1], s[1]);
            } else {
                merged.add(new long[]{s[0], s[1]});
            }
        }
        return merged;
    }

    public long screenTimeMs(int personId) {
        long total = 0;
        for (long[] s : segments(personId)) total += s[1] - s[0];
        return total;
    }

    /** Manual correction: every track of {@code fromPid} now belongs to {@code intoPid}. */
    public void merge(int fromPid, int intoPid) {
        for (int k = 0; k < personOfTrack.length; k++) if (personOfTrack[k] == fromPid) personOfTrack[k] = intoPid;
    }

    public int trackCount() { return tracks.size(); }
}
