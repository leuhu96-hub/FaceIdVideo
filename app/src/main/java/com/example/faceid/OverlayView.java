package com.example.faceid;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.media3.common.Player;

import com.example.faceid.core.VideoResult;

import java.util.Collections;
import java.util.List;

/** Draws "ID n" boxes over the video in sync with the player; tap a box to select that person. */
public class OverlayView extends View {

    public interface Listener {
        void onPersonTapped(int personId);

        void onTime(long positionMs);
    }

    private static final int[] PALETTE = {
            0xFFE6194B, 0xFF3CB44B, 0xFFFFE119, 0xFF4363D8, 0xFFF58231,
            0xFF911EB4, 0xFF42D4F4, 0xFFF032E6, 0xFFBFEF45, 0xFFFABED4,
            0xFF469990, 0xFFDCBEFF, 0xFF9A6324, 0xFFFFFAC8, 0xFFAAFFC3};

    public static int colorFor(int personId) {
        return PALETTE[Math.floorMod(personId - 1, PALETTE.length)];
    }

    private final Paint box = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBg = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF tmp = new RectF();
    private final float density;

    private Player player;
    private VideoResult result;
    private float videoAspect = 16f / 9f;
    private int selected = -1;
    private boolean onlySelected = false;
    private Listener listener;
    private List<VideoResult.Box> boxes = Collections.emptyList();
    private long lastT = -1;
    private boolean dirty = true;

    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            if (player != null) {
                long t = player.getCurrentPosition();
                if (t != lastT || dirty) {
                    lastT = t;
                    dirty = false;
                    boxes = result != null ? result.boxesAt(t) : Collections.<VideoResult.Box>emptyList();
                    if (listener != null) listener.onTime(t);
                    invalidate();
                }
            }
            postOnAnimation(this);
        }
    };

    public OverlayView(Context c, AttributeSet a) {
        super(c, a);
        density = getResources().getDisplayMetrics().density;
        box.setStyle(Paint.Style.STROKE);
        text.setColor(Color.BLACK);
        text.setTextSize(13 * getResources().getDisplayMetrics().scaledDensity);
        text.setFakeBoldText(true);
    }

    public void setPlayer(Player p) { player = p; dirty = true; }
    public void setListener(Listener l) { listener = l; }
    public void setVideoAspect(float a) { videoAspect = a; dirty = true; }
    public void setResult(VideoResult r) { result = r; dirty = true; }
    public void setSelected(int pid) { selected = pid; dirty = true; }
    public void setOnlySelected(boolean b) { onlySelected = b; dirty = true; }
    public void refresh() { dirty = true; }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        postOnAnimation(tick);
    }

    @Override
    protected void onDetachedFromWindow() {
        removeCallbacks(tick);
        super.onDetachedFromWindow();
    }

    /** Video content rect inside this view (PlayerView uses RESIZE_MODE_FIT). */
    private void contentRect(RectF out) {
        float w = getWidth(), h = getHeight();
        if (w / h > videoAspect) {
            float cw = h * videoAspect;
            out.set((w - cw) / 2, 0, (w + cw) / 2, h);
        } else {
            float ch = w / videoAspect;
            out.set(0, (h - ch) / 2, w, (h + ch) / 2);
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        if (boxes.isEmpty()) return;
        RectF cr = new RectF();
        contentRect(cr);
        float pad = 4 * density;
        for (VideoResult.Box b : boxes) {
            boolean isSel = b.personId == selected;
            if (selected > 0 && !isSel && onlySelected) continue;
            int color = colorFor(b.personId);
            int alpha = (selected > 0 && !isSel) ? 90 : 255;
            box.setColor(color);
            box.setAlpha(alpha);
            box.setStrokeWidth((isSel ? 4f : 2f) * density);
            tmp.set(cr.left + b.x0 * cr.width(), cr.top + b.y0 * cr.height(),
                    cr.left + b.x1 * cr.width(), cr.top + b.y1 * cr.height());
            canvas.drawRect(tmp, box);

            String label = "ID " + b.personId;
            float tw = text.measureText(label);
            float th = text.getTextSize();
            float top = Math.max(cr.top, tmp.top - th - pad);
            labelBg.setColor(color);
            labelBg.setAlpha(alpha);
            canvas.drawRect(tmp.left, top, tmp.left + tw + 2 * pad, top + th + pad, labelBg);
            text.setAlpha(alpha);
            canvas.drawText(label, tmp.left + pad, top + th, text);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if (e.getAction() != MotionEvent.ACTION_DOWN || boxes.isEmpty()) return false;
        RectF cr = new RectF();
        contentRect(cr);
        float x = (e.getX() - cr.left) / cr.width(), y = (e.getY() - cr.top) / cr.height();
        VideoResult.Box hit = null;
        for (VideoResult.Box b : boxes) {
            if (selected > 0 && onlySelected && b.personId != selected) continue;
            if (b.contains(x, y) && (hit == null || b.area() < hit.area())) hit = b;
        }
        if (hit == null) return false; // let PlayerView show its controls
        if (listener != null) listener.onPersonTapped(hit.personId);
        performClick();
        return true;
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }
}
