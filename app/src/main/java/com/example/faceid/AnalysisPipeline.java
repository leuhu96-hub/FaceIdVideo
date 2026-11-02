package com.example.faceid;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;

import com.example.faceid.core.FaceObs;
import com.example.faceid.core.FaceTracker;
import com.example.faceid.core.HistogramCut;
import com.example.faceid.core.Track;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** decode -> scene cut -> detect/align/embed -> track. Clustering is done afterwards (cheap, re-runnable). */
final class AnalysisPipeline {

    interface Progress {
        void onProgress(float fraction, int frames, int faces);
    }

    static final class Thumb {
        final Bitmap bmp;
        final float quality;

        Thumb(Bitmap bmp, float quality) {
            this.bmp = bmp;
            this.quality = quality;
        }
    }

    static final class Output {
        final List<Track> tracks;
        final Map<Integer, Thumb> thumbs; // by Track.id
        final float aspect;
        final long durationMs, sampleMs;
        final boolean hasEmbedder;
        final String embedderError;
        final int frames;

        Output(List<Track> tracks, Map<Integer, Thumb> thumbs, float aspect, long durationMs,
               long sampleMs, boolean hasEmbedder, String embedderError, int frames) {
            this.tracks = tracks;
            this.thumbs = thumbs;
            this.aspect = aspect;
            this.durationMs = durationMs;
            this.sampleMs = sampleMs;
            this.hasEmbedder = hasEmbedder;
            this.embedderError = embedderError;
            this.frames = frames;
        }
    }

    private final Context ctx;
    private volatile boolean cancelled;

    AnalysisPipeline(Context ctx) {
        this.ctx = ctx.getApplicationContext();
    }

    void cancel() { cancelled = true; }

    /** Blocking; run on a background thread. Returns null if cancelled. */
    Output run(Uri uri, Progress progress) throws Exception {
        final FrameExtractor.VideoInfo info = FrameExtractor.probe(ctx, uri);
        final FaceEmbedder embedder = FaceEmbedder.tryCreate(ctx, Config.EMBEDDER_ASSET);
        final FaceAnalyzer analyzer = new FaceAnalyzer(embedder);
        final FaceTracker tracker = new FaceTracker(new FaceTracker.Params());
        final HistogramCut cut = new HistogramCut(Config.SCENE_CUT_THRESHOLD);
        final Map<Integer, Thumb> thumbs = new HashMap<>();
        final float[] aspect = {0f};
        final int[] counts = {0, 0}; // frames, faces
        final int[] small = new int[64 * 36];

        try {
            new FrameExtractor().extract(ctx, uri, info.rotation, Config.SAMPLE_INTERVAL_MS,
                    Config.ANALYSIS_MAX_SIDE, (frame, ptsMs) -> {
                        if (cancelled) {
                            frame.recycle();
                            return false;
                        }
                        if (aspect[0] == 0f) aspect[0] = frame.getWidth() / (float) frame.getHeight();

                        Bitmap s = Bitmap.createScaledBitmap(frame, 64, 36, false);
                        s.getPixels(small, 0, 64, 0, 0, 64, 36);
                        s.recycle();
                        boolean sceneCut = cut.update(small);

                        List<FaceAnalyzer.Result> res = analyzer.analyze(frame, ptsMs);
                        List<FaceObs> obs = new ArrayList<>(res.size());
                        for (FaceAnalyzer.Result r : res) obs.add(r.obs);
                        int[] ids = tracker.update(ptsMs, obs, sceneCut);

                        for (int i = 0; i < res.size(); i++) {
                            FaceAnalyzer.Result r = res.get(i);
                            Thumb old = thumbs.get(ids[i]);
                            if (old == null || r.obs.quality > old.quality) {
                                if (old != null) old.bmp.recycle();
                                thumbs.put(ids[i], new Thumb(r.aligned, r.obs.quality));
                            } else {
                                r.aligned.recycle();
                            }
                        }
                        frame.recycle();
                        counts[0]++;
                        counts[1] += res.size();
                        float frac = info.durationMs > 0 ? Math.min(1f, ptsMs / (float) info.durationMs) : 0f;
                        progress.onProgress(frac, counts[0], counts[1]);
                        return !cancelled;
                    });
        } finally {
            analyzer.close();
            if (embedder != null) embedder.close();
        }
        if (cancelled) return null;

        List<Track> kept = new ArrayList<>();
        for (Track t : tracker.finish()) {
            if (t.size() >= Config.MIN_TRACK_OBS) {
                kept.add(t);
            } else {
                Thumb th = thumbs.remove(t.id);
                if (th != null) th.bmp.recycle();
            }
        }
        float a = aspect[0] > 0f ? aspect[0] : info.aspect();
        return new Output(kept, thumbs, a, info.durationMs, Config.SAMPLE_INTERVAL_MS,
                embedder != null, embedder == null ? FaceEmbedder.lastError : null, counts[0]);
    }
}
