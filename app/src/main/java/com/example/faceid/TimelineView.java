package com.example.faceid;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import java.util.Collections;
import java.util.List;

/** Shows when the selected person is on screen; tap/drag to seek. */
public class TimelineView extends View {

    public interface SeekListener {
        void onSeek(long ms);
    }

    private final Paint bg = new Paint();
    private final Paint seg = new Paint();
    private final Paint head = new Paint();
    private List<long[]> segments = Collections.emptyList();
    private long durationMs = 1, positionMs;
    private SeekListener seekListener;

    public TimelineView(Context c, AttributeSet a) {
        super(c, a);
        bg.setColor(0xFF2A2A2A);
        head.setColor(Color.WHITE);
        head.setStrokeWidth(2 * getResources().getDisplayMetrics().density);
    }

    public void setSeekListener(SeekListener l) { seekListener = l; }

    public void setDuration(long ms) {
        durationMs = Math.max(1, ms);
        invalidate();
    }

    public void setSegments(List<long[]> s, int color) {
        segments = s == null ? Collections.<long[]>emptyList() : s;
        seg.setColor(color);
        invalidate();
    }

    public void setPosition(long ms) {
        if (ms != positionMs) {
            positionMs = ms;
            invalidate();
        }
    }

    @Override
    protected void onDraw(Canvas c) {
        float w = getWidth(), h = getHeight();
        c.drawRect(0, h * 0.2f, w, h * 0.8f, bg);
        for (long[] s : segments) {
            c.drawRect(w * s[0] / durationMs, h * 0.2f, Math.max(w * s[0] / durationMs + 2, w * s[1] / durationMs),
                    h * 0.8f, seg);
        }
        float x = w * positionMs / durationMs;
        c.drawLine(x, 0, x, h, head);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        int a = e.getAction();
        if (a == MotionEvent.ACTION_DOWN || a == MotionEvent.ACTION_MOVE) {
            float f = Math.max(0f, Math.min(1f, e.getX() / getWidth()));
            if (seekListener != null) seekListener.onSeek((long) (f * durationMs));
            if (a == MotionEvent.ACTION_DOWN) performClick();
            return true;
        }
        return a == MotionEvent.ACTION_UP;
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }
}
