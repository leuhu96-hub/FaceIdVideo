package com.example.faceid.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Track-level agglomerative clustering (average linkage on cosine similarity) with
 * cannot-link constraints: two tracks seen in the same frame can never be the same person.
 * Stops when the best average similarity falls below the threshold (precision-first).
 */
public final class TrackClusterer {
    private TrackClusterer() {}

    /**
     * @return label per track (same order as {@code tracks}), 0..K-1, where label 0 is the
     *         person with the most observations. Tracks without embeddings stay singletons.
     */
    public static int[] cluster(List<Track> tracks, float threshold) {
        int n = tracks.size();
        float[][] emb = new float[n][];
        for (int i = 0; i < n; i++) emb[i] = tracks.get(i).embedding();

        boolean[][] cannot = coOccurrence(tracks);
        double[][] sum = new double[n][n];
        for (int i = 0; i < n; i++) {
            if (emb[i] == null) continue;
            for (int j = i + 1; j < n; j++) {
                if (emb[j] == null) continue;
                double s = Vec.dot(emb[i], emb[j]);
                sum[i][j] = s;
                sum[j][i] = s;
            }
        }
        int[] size = new int[n];
        boolean[] alive = new boolean[n];
        int[] clusterOf = new int[n];
        for (int i = 0; i < n; i++) {
            size[i] = 1;
            alive[i] = emb[i] != null; // only clusters with embeddings take part in merging
            clusterOf[i] = i;
        }

        while (true) {
            double best = -Double.MAX_VALUE;
            int ba = -1, bb = -1;
            for (int a = 0; a < n; a++) {
                if (!alive[a]) continue;
                for (int b = a + 1; b < n; b++) {
                    if (!alive[b] || cannot[a][b]) continue;
                    double avg = sum[a][b] / ((double) size[a] * size[b]);
                    if (avg > best) {
                        best = avg;
                        ba = a;
                        bb = b;
                    }
                }
            }
            if (ba < 0 || best < threshold) break;
            // merge bb into ba
            for (int k = 0; k < n; k++) {
                if (k == ba || k == bb) continue;
                double s = sum[ba][k] + sum[bb][k];
                sum[ba][k] = s;
                sum[k][ba] = s;
                boolean c = cannot[ba][k] || cannot[bb][k];
                cannot[ba][k] = c;
                cannot[k][ba] = c;
            }
            size[ba] += size[bb];
            alive[bb] = false;
            for (int i = 0; i < n; i++) if (clusterOf[i] == bb) clusterOf[i] = ba;
        }
        return renumberByCount(tracks, clusterOf);
    }

    /** Labels 0..K-1 ordered by total number of observations (descending). */
    public static int[] renumberByCount(List<Track> tracks, int[] clusterOf) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int i = 0; i < clusterOf.length; i++) {
            Integer c = count.get(clusterOf[i]);
            count.put(clusterOf[i], (c == null ? 0 : c) + tracks.get(i).size());
        }
        List<Map.Entry<Integer, Integer>> entries = new ArrayList<>(count.entrySet());
        entries.sort((x, y) -> {
            int c = Integer.compare(y.getValue(), x.getValue());
            return c != 0 ? c : Integer.compare(x.getKey(), y.getKey());
        });
        Map<Integer, Integer> label = new HashMap<>();
        for (int i = 0; i < entries.size(); i++) label.put(entries.get(i).getKey(), i);
        int[] out = new int[clusterOf.length];
        for (int i = 0; i < clusterOf.length; i++) out[i] = label.get(clusterOf[i]);
        return out;
    }

    static boolean[][] coOccurrence(List<Track> tracks) {
        int n = tracks.size();
        boolean[][] c = new boolean[n][n];
        Map<Long, List<Integer>> byTime = new HashMap<>();
        for (int i = 0; i < n; i++) {
            for (FaceObs o : tracks.get(i).obs) {
                List<Integer> l = byTime.get(o.tMs);
                if (l == null) {
                    l = new ArrayList<>();
                    byTime.put(o.tMs, l);
                }
                l.add(i);
            }
        }
        for (List<Integer> l : byTime.values()) {
            for (int a = 0; a < l.size(); a++) {
                for (int b = a + 1; b < l.size(); b++) {
                    int i = l.get(a), j = l.get(b);
                    if (i == j) continue;
                    c[i][j] = true;
                    c[j][i] = true;
                }
            }
        }
        return c;
    }
}
