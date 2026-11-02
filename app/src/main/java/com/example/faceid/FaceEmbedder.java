package com.example.faceid;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import com.example.faceid.core.Vec;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Face embedding with any float32 TFLite model: input [1,H,W,3] (NHWC) or [1,3,H,W] (NCHW),
 * output with D elements (e.g. 128/192/512). Shapes are read from the model.
 */
final class FaceEmbedder implements Closeable {
    private static final String TAG = "FaceEmbedder";
    static String lastError;

    private final Interpreter tflite;
    private final int inW, inH, dim;
    private final boolean nchw;
    private final ByteBuffer in, out;
    private final int[] px;

    static FaceEmbedder tryCreate(Context ctx, String asset) {
        try {
            return new FaceEmbedder(load(ctx, asset));
        } catch (IOException e) {
            lastError = "Không có model " + asset + " trong assets";
        } catch (RuntimeException e) {
            lastError = "Không nạp được model: " + e.getMessage();
        }
        Log.w(TAG, lastError);
        return null;
    }

    private static MappedByteBuffer load(Context ctx, String asset) throws IOException {
        try (AssetFileDescriptor afd = ctx.getAssets().openFd(asset);
             FileInputStream fis = new FileInputStream(afd.getFileDescriptor())) {
            return fis.getChannel().map(FileChannel.MapMode.READ_ONLY,
                    afd.getStartOffset(), afd.getDeclaredLength());
        }
    }

    private FaceEmbedder(MappedByteBuffer model) {
        Interpreter.Options o = new Interpreter.Options();
        o.setNumThreads(Config.EMBEDDER_THREADS);
        tflite = new Interpreter(model, o);
        Tensor t = tflite.getInputTensor(0);
        if (t.dataType() != DataType.FLOAT32) {
            tflite.close();
            throw new IllegalArgumentException("input phải là float32 (đang là " + t.dataType() + ")");
        }
        int[] s = t.shape();
        if (s.length == 4 && s[3] == 3) {
            nchw = false;
            inH = s[1];
            inW = s[2];
        } else if (s.length == 4 && s[1] == 3) {
            nchw = true;
            inH = s[2];
            inW = s[3];
        } else {
            tflite.close();
            throw new IllegalArgumentException("shape input không hỗ trợ");
        }
        dim = tflite.getOutputTensor(0).numElements();
        in = ByteBuffer.allocateDirect(4 * inW * inH * 3).order(ByteOrder.nativeOrder());
        out = ByteBuffer.allocateDirect(4 * dim).order(ByteOrder.nativeOrder());
        px = new int[inW * inH];
        Log.i(TAG, "model " + inW + "x" + inH + (nchw ? " NCHW" : " NHWC") + " -> " + dim);
    }

    /** @return L2-normalized embedding. */
    synchronized float[] embed(Bitmap face) {
        Bitmap src = (face.getWidth() == inW && face.getHeight() == inH)
                ? face : Bitmap.createScaledBitmap(face, inW, inH, true);
        src.getPixels(px, 0, inW, 0, 0, inW, inH);
        if (src != face) src.recycle();

        in.rewind();
        FloatBuffer fb = in.asFloatBuffer();
        final float m = Config.INPUT_MEAN, sd = Config.INPUT_STD;
        if (!nchw) {
            for (int c : px) {
                fb.put((((c >> 16) & 0xff) - m) / sd);
                fb.put((((c >> 8) & 0xff) - m) / sd);
                fb.put(((c & 0xff) - m) / sd);
            }
        } else {
            for (int shift = 16; shift >= 0; shift -= 8) {
                for (int c : px) fb.put((((c >> shift) & 0xff) - m) / sd);
            }
        }
        out.rewind();
        tflite.run(in, out);
        out.rewind();
        float[] e = new float[dim];
        out.asFloatBuffer().get(e);
        return Vec.normalize(e);
    }

    @Override
    public void close() {
        tflite.close();
    }
}
