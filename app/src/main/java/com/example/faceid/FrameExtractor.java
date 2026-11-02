package com.example.faceid;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.net.Uri;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Sequential decode with MediaCodec (much faster than MediaMetadataRetriever.getFrameAtTime per frame),
 * emitting one downscaled RGB frame every {@code intervalMs} of presentation time.
 */
final class FrameExtractor {

    interface Callback {
        /** @return false to stop decoding. The callback owns the bitmap. */
        boolean onFrame(Bitmap frame, long ptsMs);
    }

    static final class VideoInfo {
        final long durationMs;
        final int rotation, width, height;

        VideoInfo(long durationMs, int rotation, int width, int height) {
            this.durationMs = durationMs;
            this.rotation = rotation;
            this.width = width;
            this.height = height;
        }

        /** Display aspect ratio after rotation. */
        float aspect() {
            if (width <= 0 || height <= 0) return 16f / 9f;
            boolean swap = rotation == 90 || rotation == 270;
            return swap ? height / (float) width : width / (float) height;
        }
    }

    static VideoInfo probe(Context ctx, Uri uri) throws IOException {
        MediaMetadataRetriever r = new MediaMetadataRetriever();
        try {
            r.setDataSource(ctx, uri);
            return new VideoInfo(
                    parseLong(r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)),
                    (int) parseLong(r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)),
                    (int) parseLong(r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)),
                    (int) parseLong(r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)));
        } finally {
            r.release();
        }
    }

    private static long parseLong(String s) {
        try {
            return s == null ? 0 : Long.parseLong(s.trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    void extract(Context ctx, Uri uri, int rotation, long intervalMs, int maxSide, Callback cb) throws IOException {
        MediaExtractor ex = new MediaExtractor();
        MediaCodec codec = null;
        try {
            ex.setDataSource(ctx, uri, null);
            int track = -1;
            MediaFormat fmt = null;
            for (int i = 0; i < ex.getTrackCount(); i++) {
                MediaFormat f = ex.getTrackFormat(i);
                String mime = f.getString(MediaFormat.KEY_MIME);
                if (mime != null && mime.startsWith("video/")) {
                    track = i;
                    fmt = f;
                    break;
                }
            }
            if (track < 0) throw new IOException("Không tìm thấy track video");
            ex.selectTrack(track);
            String mime = fmt.getString(MediaFormat.KEY_MIME);
            codec = MediaCodec.createDecoderByType(mime);
            fmt.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            codec.configure(fmt, null, null, 0);
            codec.start();

            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            long intervalUs = intervalMs * 1000;
            long nextUs = 0;
            boolean inEos = false, outEos = false, stop = false;
            while (!outEos && !stop) {
                if (!inEos) {
                    int in = codec.dequeueInputBuffer(10_000);
                    if (in >= 0) {
                        ByteBuffer buf = codec.getInputBuffer(in);
                        int n = buf == null ? -1 : ex.readSampleData(buf, 0);
                        if (n < 0) {
                            codec.queueInputBuffer(in, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inEos = true;
                        } else {
                            codec.queueInputBuffer(in, 0, n, ex.getSampleTime(), 0);
                            ex.advance();
                        }
                    }
                }
                int out = codec.dequeueOutputBuffer(info, 10_000);
                if (out >= 0) {
                    if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) outEos = true;
                    if (info.size > 0 && info.presentationTimeUs >= nextUs) {
                        Image img = codec.getOutputImage(out);
                        if (img != null) {
                            Bitmap bmp;
                            try {
                                bmp = YuvConverter.toBitmap(img, maxSide, rotation);
                            } finally {
                                img.close();
                            }
                            while (nextUs <= info.presentationTimeUs) nextUs += intervalUs;
                            if (!cb.onFrame(bmp, info.presentationTimeUs / 1000)) stop = true;
                        }
                    }
                    codec.releaseOutputBuffer(out, false);
                }
            }
        } finally {
            if (codec != null) {
                try {
                    codec.stop();
                } catch (IllegalStateException ignored) {
                }
                codec.release();
            }
            ex.release();
        }
    }
}
