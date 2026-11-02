package com.example.faceid;

/** Tunable parameters. */
public final class Config {
    private Config() {}

    /** Analysis sampling interval (~6 fps). Smaller = more accurate tracking, slower analysis. */
    public static final long SAMPLE_INTERVAL_MS = 166;
    /** Decoded frames are downscaled so that the longest side is at most this. */
    public static final int ANALYSIS_MAX_SIDE = 960;

    /** Face embedding model in app/src/main/assets. If missing, the app runs in tracking-only mode. */
    public static final String EMBEDDER_ASSET = "face_embedder.tflite";
    /** Input normalization: (pixel - MEAN) / STD, RGB. EdgeFace/ArcFace: 127.5 / 127.5. MobileFaceNet: 127.5 / 128. */
    public static final float INPUT_MEAN = 127.5f;
    public static final float INPUT_STD = 127.5f;
    public static final int EMBEDDER_THREADS = 4;

    /** Faces smaller than this (pixels, short side, in the analysis frame) get no embedding. */
    public static final int MIN_EMBED_FACE_PX = 40;
    /** Faces with quality below this get no embedding (still tracked). */
    public static final float MIN_EMBED_QUALITY = 0.25f;
    /** Face size (px) considered "fully good" for the quality score. */
    public static final float GOOD_FACE_PX = 96f;

    /** Default cosine-similarity threshold for merging tracks into one person (adjustable in the UI). */
    public static final float DEFAULT_CLUSTER_THRESHOLD = 0.45f;
    public static final float SCENE_CUT_THRESHOLD = 0.35f;
    /** Tracks with fewer observations are dropped as noise. */
    public static final int MIN_TRACK_OBS = 2;
}
