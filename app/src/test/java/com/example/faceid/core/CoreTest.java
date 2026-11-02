package com.example.faceid.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

public class CoreTest {

    private static float[] noisy(float[] base, Random r, float noise) {
        float[] v = new float[base.length];
        for (int i = 0; i < v.length; i++) v[i] = base[i] + noise * (float) r.nextGaussian();
        return Vec.normalize(v);
    }

    private static float[] randomUnit(Random r, int d) {
        float[] v = new float[d];
        for (int i = 0; i < d; i++) v[i] = (float) r.nextGaussian();
        return Vec.normalize(v);
    }

    /** Two people, A appears in shot 1 and shot 3, B in shots 1 and 2; A and B co-occur in shot 1. */
    @Test
    public void tracksAndClustersTwoPeopleAcrossScenes() {
        Random r = new Random(1);
        float[] A = randomUnit(r, 128), B = randomUnit(r, 128);
        FaceTracker tr = new FaceTracker(new FaceTracker.Params());
        long t = 0;
        boolean cut;
        // shot 1: A left, B right, slowly moving
        for (int i = 0; i < 20; i++, t += 166) {
            float dx = i * 0.003f;
            List<FaceObs> d = Arrays.asList(
                    new FaceObs(t, 0.1f + dx, 0.2f, 0.3f + dx, 0.5f, noisy(A, r, 0.03f), 0.9f),
                    new FaceObs(t, 0.6f - dx, 0.2f, 0.8f - dx, 0.5f, noisy(B, r, 0.03f), 0.9f));
            tr.update(t, d, false);
        }
        // shot 2 (cut): only B, at A's old position
        for (int i = 0; i < 10; i++, t += 166) {
            cut = i == 0;
            tr.update(t, Arrays.asList(new FaceObs(t, 0.1f, 0.2f, 0.3f, 0.5f, noisy(B, r, 0.03f), 0.9f)), cut);
        }
        // shot 3 (cut): only A, at the same position
        for (int i = 0; i < 10; i++, t += 166) {
            cut = i == 0;
            tr.update(t, Arrays.asList(new FaceObs(t, 0.1f, 0.2f, 0.3f, 0.5f, noisy(A, r, 0.03f), 0.9f)), cut);
        }
        List<Track> tracks = tr.finish();
        assertEquals("one track per person per shot", 4, tracks.size());

        int[] labels = TrackClusterer.cluster(tracks, 0.45f);
        // track order by id: A1, B1, B2, A3
        assertEquals(labels[0], labels[3]);
        assertEquals(labels[1], labels[2]);
        assertNotEquals(labels[0], labels[1]);

        VideoResult vr = new VideoResult(tracks, labels, 166);
        assertEquals(2, vr.personIds().size());
        assertEquals(2, vr.boxesAt(166 * 5 + 50).size());
        assertEquals(1, vr.boxesAt(166 * 25).size());
    }

    @Test
    public void cannotLinkPreventsMergingCoOccurringTracks() {
        Random r = new Random(2);
        float[] A = randomUnit(r, 64);
        List<Track> tracks = new ArrayList<>();
        Track t1 = new Track(0), t2 = new Track(1);
        for (int i = 0; i < 5; i++) {
            t1.add(new FaceObs(i * 100, 0f, 0f, .1f, .1f, noisy(A, r, 0.01f), 1f));
            t2.add(new FaceObs(i * 100, .5f, .5f, .6f, .6f, noisy(A, r, 0.01f), 1f)); // identical look-alike
        }
        tracks.add(t1);
        tracks.add(t2);
        int[] labels = TrackClusterer.cluster(tracks, 0.3f);
        assertNotEquals(labels[0], labels[1]);
    }

    @Test
    public void similarityTransformRecoversKnownTransform() {
        float a = 0.8f * (float) Math.cos(0.3), b = 0.8f * (float) Math.sin(0.3), tx = 5f, ty = -3f;
        float[] src = {10, 20, 40, 22, 25, 35, 12, 50, 38, 52};
        float[] dst = new float[10];
        for (int i = 0; i < 5; i++) {
            float x = src[2 * i], y = src[2 * i + 1];
            dst[2 * i] = a * x - b * y + tx;
            dst[2 * i + 1] = b * x + a * y + ty;
        }
        float[] t = SimilarityTransform.estimate(src, dst);
        assertEquals(a, t[0], 1e-4);
        assertEquals(b, t[1], 1e-4);
        assertEquals(tx, t[2], 1e-3);
        assertEquals(ty, t[3], 1e-3);
    }

    @Test
    public void histogramCutDetectsChange() {
        HistogramCut c = new HistogramCut(0.35f);
        int[] dark = new int[100], bright = new int[100];
        Arrays.fill(dark, 0xff101010);
        Arrays.fill(bright, 0xffe0e0e0);
        assertTrue(!c.update(dark));
        assertTrue(!c.update(dark));
        assertTrue(c.update(bright));
    }
}
