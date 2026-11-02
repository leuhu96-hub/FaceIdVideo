package com.example.faceid;

import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.media3.common.MediaItem;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.media3.ui.PlayerView;

import com.example.faceid.core.TrackClusterer;
import com.example.faceid.core.VideoResult;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements OverlayView.Listener {

    private ExoPlayer player;
    private OverlayView overlay;
    private TimelineView timeline;
    private TextView status, thresholdLabel;
    private ProgressBar progress;
    private SeekBar thresholdBar;
    private LinearLayout personsContainer;
    private CheckBox onlySelected;
    private Button mergeBtn;

    private final ExecutorService exec = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private AnalysisPipeline pipeline;
    private AnalysisPipeline.Output analysis;
    private VideoResult result;
    private int selectedPid = -1;
    private boolean mergeMode = false;
    private float threshold = Config.DEFAULT_CLUSTER_THRESHOLD;

    private final ActivityResultLauncher<PickVisualMediaRequest> picker =
            registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
                if (uri != null) openVideo(uri);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        PlayerView playerView = findViewById(R.id.player_view);
        overlay = findViewById(R.id.overlay);
        timeline = findViewById(R.id.timeline);
        status = findViewById(R.id.status);
        progress = findViewById(R.id.progress);
        thresholdBar = findViewById(R.id.threshold);
        thresholdLabel = findViewById(R.id.threshold_label);
        personsContainer = findViewById(R.id.persons);
        onlySelected = findViewById(R.id.only_selected);
        mergeBtn = findViewById(R.id.merge);

        player = new ExoPlayer.Builder(this).build();
        player.setRepeatMode(ExoPlayer.REPEAT_MODE_ONE);
        playerView.setPlayer(player);
        overlay.setPlayer(player);
        overlay.setListener(this);
        timeline.setSeekListener(ms -> player.seekTo(ms));

        findViewById(R.id.pick).setOnClickListener(v -> picker.launch(new PickVisualMediaRequest.Builder()
                .setMediaType(ActivityResultContracts.PickVisualMedia.VideoOnly.INSTANCE).build()));
        onlySelected.setOnCheckedChangeListener((b, checked) -> overlay.setOnlySelected(checked));
        mergeBtn.setOnClickListener(v -> {
            if (selectedPid <= 0) return;
            mergeMode = !mergeMode;
            updateSelectionUi();
        });

        thresholdBar.setMax(60); // 0.20 .. 0.80
        thresholdBar.setProgress(Math.round((threshold - 0.20f) * 100));
        updateThresholdLabel();
        thresholdBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar s, int p, boolean fromUser) {
                threshold = 0.20f + p / 100f;
                updateThresholdLabel();
            }

            @Override
            public void onStartTrackingTouch(SeekBar s) {}

            @Override
            public void onStopTrackingTouch(SeekBar s) {
                if (analysis != null) recluster();
            }
        });
        status.setText(R.string.hint_pick);
    }

    private void updateThresholdLabel() {
        thresholdLabel.setText(getString(R.string.threshold_label, threshold));
    }

    private void openVideo(Uri uri) {
        if (pipeline != null) pipeline.cancel();
        analysis = null;
        result = null;
        selectedPid = -1;
        mergeMode = false;
        overlay.setResult(null);
        overlay.setSelected(-1);
        personsContainer.removeAllViews();
        timeline.setSegments(null, 0);
        mergeBtn.setEnabled(false);

        player.setMediaItem(MediaItem.fromUri(uri));
        player.prepare();
        player.pause();

        progress.setVisibility(View.VISIBLE);
        progress.setProgress(0);
        status.setText(R.string.analyzing);

        final AnalysisPipeline p = new AnalysisPipeline(this);
        pipeline = p;
        exec.execute(() -> {
            try {
                AnalysisPipeline.Output out = p.run(uri, (frac, frames, faces) -> main.post(() -> {
                    if (pipeline != p) return;
                    progress.setProgress(Math.round(frac * 1000));
                    status.setText(getString(R.string.analyzing_progress, Math.round(frac * 100), frames, faces));
                }));
                if (out != null) main.post(() -> {
                    if (pipeline == p) onAnalysisDone(out);
                });
            } catch (Throwable e) {
                main.post(() -> {
                    if (pipeline != p) return;
                    progress.setVisibility(View.GONE);
                    status.setText(getString(R.string.error, String.valueOf(e.getMessage())));
                });
            }
        });
    }

    private void onAnalysisDone(AnalysisPipeline.Output out) {
        analysis = out;
        progress.setVisibility(View.GONE);
        overlay.setVideoAspect(out.aspect);
        timeline.setDuration(out.durationMs > 0 ? out.durationMs : player.getDuration());
        recluster();
        if (!out.hasEmbedder) {
            Toast.makeText(this, getString(R.string.no_model, out.embedderError), Toast.LENGTH_LONG).show();
        }
        player.play();
    }

    /** Re-runs track clustering with the current threshold (manual merges are reset). */
    private void recluster() {
        // Without a model nothing is merged: threshold > 1 keeps every track as its own person.
        float th = analysis.hasEmbedder ? threshold : 2f;
        int[] labels = TrackClusterer.cluster(analysis.tracks, th);
        result = new VideoResult(analysis.tracks, labels, analysis.sampleMs);
        overlay.setResult(result);
        selectedPid = -1;
        mergeMode = false;
        rebuildChips();
        updateSelectionUi();
    }

    private void rebuildChips() {
        personsContainer.removeAllViews();
        if (result == null) return;
        int thumbPx = dp(60);
        for (int pid : result.personIds()) {
            LinearLayout chip = new LinearLayout(this);
            chip.setOrientation(LinearLayout.VERTICAL);
            chip.setGravity(Gravity.CENTER_HORIZONTAL);
            chip.setPadding(dp(4), dp(4), dp(4), dp(4));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
            lp.setMarginEnd(dp(6));
            chip.setLayoutParams(lp);

            ImageView iv = new ImageView(this);
            iv.setLayoutParams(new LinearLayout.LayoutParams(thumbPx, thumbPx));
            AnalysisPipeline.Thumb best = bestThumb(pid);
            if (best != null) iv.setImageBitmap(best.bmp);
            chip.addView(iv);

            TextView tv = new TextView(this);
            tv.setGravity(Gravity.CENTER);
            tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
            tv.setText(String.format(Locale.US, "ID %d\n%.1fs", pid, result.screenTimeMs(pid) / 1000f));
            chip.addView(tv);

            GradientDrawable bg = new GradientDrawable();
            bg.setCornerRadius(dp(8));
            bg.setStroke(dp(pid == selectedPid ? 3 : 1), OverlayView.colorFor(pid));
            if (pid == selectedPid) bg.setColor(0x33FFFFFF);
            chip.setBackground(bg);
            chip.setOnClickListener(v -> onPersonTapped(pid));
            personsContainer.addView(chip);
        }
    }

    private AnalysisPipeline.Thumb bestThumb(int pid) {
        AnalysisPipeline.Thumb best = null;
        for (int tid : result.trackIdsOf(pid)) {
            AnalysisPipeline.Thumb t = analysis.thumbs.get(tid);
            if (t != null && (best == null || t.quality > best.quality)) best = t;
        }
        return best;
    }

    @Override
    public void onPersonTapped(int pid) {
        if (result == null) return;
        if (mergeMode && selectedPid > 0 && pid != selectedPid) {
            result.merge(pid, selectedPid);
            mergeMode = false;
            Toast.makeText(this, getString(R.string.merged, pid, selectedPid), Toast.LENGTH_SHORT).show();
        } else {
            mergeMode = false;
            selectedPid = (pid == selectedPid) ? -1 : pid;
        }
        overlay.refresh();
        rebuildChips();
        updateSelectionUi();
    }

    @Override
    public void onTime(long positionMs) {
        timeline.setPosition(positionMs);
    }

    private void updateSelectionUi() {
        overlay.setSelected(selectedPid);
        mergeBtn.setEnabled(selectedPid > 0);
        mergeBtn.setText(mergeMode ? R.string.merge_cancel : R.string.merge);
        if (result == null) return;
        if (selectedPid > 0) {
            List<long[]> segs = result.segments(selectedPid);
            timeline.setSegments(segs, OverlayView.colorFor(selectedPid));
            if (mergeMode) {
                status.setText(getString(R.string.merge_hint, selectedPid));
            } else {
                status.setText(getString(R.string.selected_info, selectedPid, segs.size(),
                        result.screenTimeMs(selectedPid) / 1000f));
            }
        } else {
            timeline.setSegments(null, 0);
            status.setText(getString(R.string.done_info, result.personIds().size(), result.trackCount(),
                    analysis.frames));
        }
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onStop() {
        super.onStop();
        player.pause();
    }

    @Override
    protected void onDestroy() {
        if (pipeline != null) pipeline.cancel();
        exec.shutdownNow();
        player.release();
        super.onDestroy();
    }
}
