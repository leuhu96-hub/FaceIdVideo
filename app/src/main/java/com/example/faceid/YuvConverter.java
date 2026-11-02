package com.example.faceid;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.media.Image;

import java.nio.ByteBuffer;

/** YUV_420_888 Image -> downscaled, rotated ARGB Bitmap (BT.601). */
final class YuvConverter {
    private YuvConverter() {}

    static Bitmap toBitmap(Image img, int maxSide, int rotationDeg) {
        Rect crop = img.getCropRect();
        int w = crop.width(), h = crop.height();
        int step = Math.max(1, (int) Math.ceil(Math.max(w, h) / (double) maxSide));
        int ow = w / step, oh = h / step;

        Image.Plane[] p = img.getPlanes();
        ByteBuffer yb = p[0].getBuffer(), ub = p[1].getBuffer(), vb = p[2].getBuffer();
        int yr = p[0].getRowStride(), yp = p[0].getPixelStride();
        int ur = p[1].getRowStride(), up = p[1].getPixelStride();
        int vr = p[2].getRowStride(), vp = p[2].getPixelStride();

        int[] out = new int[ow * oh];
        for (int oy = 0; oy < oh; oy++) {
            int y = crop.top + oy * step;
            int yRow = y * yr;
            int uRow = (y >> 1) * ur;
            int vRow = (y >> 1) * vr;
            int o = oy * ow;
            for (int ox = 0; ox < ow; ox++) {
                int x = crop.left + ox * step;
                int Y = (yb.get(yRow + x * yp) & 0xff) - 16;
                int U = (ub.get(uRow + (x >> 1) * up) & 0xff) - 128;
                int V = (vb.get(vRow + (x >> 1) * vp) & 0xff) - 128;
                if (Y < 0) Y = 0;
                int c = 1192 * Y;
                int r = clamp(c + 1634 * V);
                int g = clamp(c - 833 * V - 400 * U);
                int b = clamp(c + 2066 * U);
                out[o + ox] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }
        Bitmap bmp = Bitmap.createBitmap(out, ow, oh, Bitmap.Config.ARGB_8888);
        if (rotationDeg % 360 != 0) {
            Matrix m = new Matrix();
            m.postRotate(rotationDeg);
            Bitmap rot = Bitmap.createBitmap(bmp, 0, 0, ow, oh, m, true);
            if (rot != bmp) bmp.recycle();
            return rot;
        }
        return bmp;
    }

    private static int clamp(int v) {
        if (v < 0) return 0;
        if (v > 262143) return 255;
        return v >> 10;
    }
}
