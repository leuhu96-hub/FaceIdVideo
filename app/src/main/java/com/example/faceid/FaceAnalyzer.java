package com.example.faceid;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;

import com.example.faceid.core.FaceObs;
import com.example.faceid.core.SimilarityTransform;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

/** Detect (ML Kit) -> 5-point align to 112x112 (ArcFace template) -> quality -> embed. */
final class FaceAnalyzer implements Closeable {

    static final class Result {
        final FaceObs obs;
        final Bitmap aligned;

        Result(FaceObs obs, Bitmap aligned) {
            this.obs = obs;
            this.aligned = aligned;
        }
    }

    static final int ALIGN_SIZE = 112;
    /** ArcFace 112x112 template: left eye, right eye, nose, left mouth, right mouth (image coords). */
    private static final float[] TEMPLATE = {
            38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f, 71.7366f,
            41.5493f, 92.3655f, 70.7299f, 92.2041f};
    private static final int[] LANDMARKS = {
            FaceLandmark.LEFT_EYE, FaceLandmark.RIGHT_EYE, FaceLandmark.NOSE_BASE,
            FaceLandmark.MOUTH_LEFT, FaceLandmark.MOUTH_RIGHT};

    private final FaceDetector detector;
    private final FaceEmbedder embedder;
    private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG | Paint.ANTI_ALIAS_FLAG);

    FaceAnalyzer(FaceEmbedder embedder) {
        this.embedder = embedder;
        FaceDetectorOptions o = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setMinFaceSize(0.04f)
                .build();
        detector = FaceDetection.getClient(o);
    }

    /** Must be called off the main thread. */
    List<Result> analyze(Bitmap frame, long tMs) {
        List<Face> faces;
        try {
            faces = Tasks.await(detector.process(InputImage.fromBitmap(frame, 0)));
        } catch (ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        }
        int W = frame.getWidth(), H = frame.getHeight();
        List<Result> out = new ArrayList<>(faces.size());
        for (Face f : faces) {
            Rect b = f.getBoundingBox();
            float x0 = clamp(b.left, 0, W), y0 = clamp(b.top, 0, H);
            float x1 = clamp(b.right, 0, W), y1 = clamp(b.bottom, 0, H);
            if (x1 - x0 < 8 || y1 - y0 < 8) continue;

            float size = Math.min(b.width(), b.height());
            float[] lm = landmarks(f);
            float frontal = (float) Math.max(0.0, Math.cos(Math.toRadians(f.getHeadEulerAngleY())));
            float q = Math.min(1f, size / Config.GOOD_FACE_PX) * frontal * (lm != null ? 1f : 0.5f);

            Bitmap aligned = align(frame, lm, b);
            float[] emb = null;
            if (embedder != null && size >= Config.MIN_EMBED_FACE_PX && q >= Config.MIN_EMBED_QUALITY) {
                emb = embedder.embed(aligned);
            }
            out.add(new Result(new FaceObs(tMs, x0 / W, y0 / H, x1 / W, y1 / H, emb, q), aligned));
        }
        return out;
    }

    /** 5 landmarks ordered as the template (image-left eye first), or null if any is missing. */
    private static float[] landmarks(Face f) {
        float[] p = new float[10];
        for (int i = 0; i < LANDMARKS.length; i++) {
            FaceLandmark l = f.getLandmark(LANDMARKS[i]);
            if (l == null) return null;
            PointF pt = l.getPosition();
            p[2 * i] = pt.x;
            p[2 * i + 1] = pt.y;
        }
        // ML Kit names landmarks from the subject's point of view; the template is in image space.
        sortPairByX(p, 0, 1);
        sortPairByX(p, 3, 4);
        return p;
    }

    private static void sortPairByX(float[] p, int i, int j) {
        if (p[2 * i] > p[2 * j]) {
            float tx = p[2 * i], ty = p[2 * i + 1];
            p[2 * i] = p[2 * j];
            p[2 * i + 1] = p[2 * j + 1];
            p[2 * j] = tx;
            p[2 * j + 1] = ty;
        }
    }

    private Bitmap align(Bitmap frame, float[] lm, Rect box) {
        Bitmap out = Bitmap.createBitmap(ALIGN_SIZE, ALIGN_SIZE, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(out);
        c.drawColor(Color.BLACK);
        Matrix m = new Matrix();
        if (lm != null) {
            float[] t = SimilarityTransform.estimate(lm, TEMPLATE);
            m.setValues(new float[]{t[0], -t[1], t[2], t[1], t[0], t[3], 0f, 0f, 1f});
        } else {
            float cx = box.exactCenterX(), cy = box.exactCenterY();
            float s = Math.max(box.width(), box.height());
            m.setRectToRect(new RectF(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2),
                    new RectF(0, 0, ALIGN_SIZE, ALIGN_SIZE), Matrix.ScaleToFit.FILL);
        }
        c.drawBitmap(frame, m, paint);
        return out;
    }

    private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }

    @Override
    public void close() {
        detector.close();
    }
}
