#!/usr/bin/env python3
"""Đánh giá che mặt: đo tỷ lệ lộ mặt của kết quả che so với GT mở rộng.

Cùng công thức với nút "Đánh giá che mặt" trong Mini CVAT (index.html):
  - GT: file "JSONL che mặt" xuất từ Mini CVAT (dòng đầu là meta có target_ids).
  - Kết quả: vùng bị che ở từng frame, JSONL (sample_id + object[bbox, score?]) hoặc MOT txt
    (frame,id,x,y,w,h[,score], frame đếm từ 1).
  - Mặt của người KHÔNG được chọn bị tính là "lộ" khi vùng che phủ dưới --cover diện tích box mặt GT.
  - Mặt có ignore=true không tính. Frame không có trong file kết quả = không che gì.

Chỉ dùng thư viện chuẩn của Python.

  python scripts/eval_privacy.py --gt clip_privacy_gt.jsonl --pred clip_masks.jsonl --out res
  python scripts/eval_privacy.py --gt-dir gt --pred-dir masks --out res      # ghép theo tên file
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

SIZES = [(0, 16, "<16 px"), (16, 32, "16–32 px"), (32, 64, "32–64 px"), (64, 1e9, "≥64 px")]
ENUMS = {
    "blur": ("Độ mờ", ["rõ", "mờ vừa", "mờ nặng"]),
    "occlusion": ("Bị che", ["không", "một phần", "nặng"]),
    "pose": ("Góc mặt", ["thẳng", "nghiêng", "ngang"]),
}
IDENT = ["yes", "unsure", "no"]
IDENT_VI = ["có", "không chắc", "không"]
SUFFIX = re.compile(r"(_privacy_gt|_gt|_masks?|_pred|_result|_results|_blur|_out)$", re.I)


def frame_of(rec):
    for k in ("frame", "frame_id", "frame_idx", "frame_no"):
        if k in rec:
            return int(rec[k])
    m = re.findall(r"\d+", str(rec.get("sample_id", "")).rsplit("/", 1)[-1])
    return int(m[-1]) if m else None


def to_xywh(b, fmt):
    a, c, d, e = (float(v) for v in b[:4])
    if fmt == "xyxy":
        return a, c, d - a, e - c
    if fmt == "cxcywh":
        return a - d / 2, c - e / 2, d, e
    return a, c, d, e


def load_gt(path):
    """-> (meta, {frame: [obj]}); obj có x,y,w,h,cls,track,person,blur,occlusion,pose,ident,indirect,truncated,ignore"""
    meta, frames = {}, {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "meta" in rec and "sample_id" not in rec:
            meta = rec["meta"]
            continue
        f = frame_of(rec)
        if f is None:
            continue
        f -= int(meta.get("frame_base", 0))
        objs = frames.setdefault(f, [])
        for o in rec.get("object") or rec.get("objects") or []:
            x, y, w, h = to_xywh(o["bbox"], meta.get("bbox", "xywh"))
            if w <= 0 or h <= 0:
                continue
            ident = o.get("identifiable", "yes")
            ident = IDENT.index(ident) if ident in IDENT else (0 if ident is True else 2 if ident is False else 0)
            track = o.get("track_id")
            objs.append({
                "x": x, "y": y, "w": w, "h": h,
                "cls": str(o.get("cls") or o.get("label") or "face"),
                "track": track, "person": o.get("person_id", track),
                "blur": int(o.get("blur", 0)), "occlusion": int(o.get("occlusion", 0)), "pose": int(o.get("pose", 0)),
                "ident": ident, "indirect": bool(o.get("indirect", False)),
                "truncated": bool(o.get("truncated", False)), "ignore": bool(o.get("ignore", False)),
            })
    return meta, frames


def load_pred(path, fmt, offset, width=None, height=None, normalized=False):
    """-> {frame: [{x,y,w,h,s}]}"""
    out = {}
    text = Path(path).read_text(encoding="utf-8-sig")
    first = text.lstrip()[:1]
    if first in "{[":
        recs = json.loads(text) if first == "[" else [json.loads(l) for l in text.splitlines() if l.strip()]
        for rec in recs:
            if "meta" in rec and "sample_id" not in rec:
                continue
            f = frame_of(rec)
            if f is None:
                continue
            L = out.setdefault(f - offset, [])
            for o in rec.get("object") or rec.get("objects") or rec.get("faces") or rec.get("detections") or []:
                b = o.get("bbox") or o.get("box")
                if not b:
                    continue
                if normalized:
                    b = [b[0] * width, b[1] * height, b[2] * width, b[3] * height]
                x, y, w, h = to_xywh(b, fmt)
                s = o.get("score", o.get("confidence", o.get("conf", 1.0)))
                if w > 0 and h > 0:
                    L.append({"x": x, "y": y, "w": w, "h": h, "s": float(s)})
    else:  # MOT: frame,id,x,y,w,h[,score] – frame đếm từ 1
        for line in text.splitlines():
            v = [t for t in re.split(r"[,\s]+", line.strip()) if t]
            if len(v) < 6:
                continue
            try:
                v = [float(t) for t in v]
            except ValueError:
                continue
            s = v[6] if len(v) > 6 and 0 <= v[6] <= 1 else 1.0
            out.setdefault(int(round(v[0])) - 1, []).append({"x": v[2], "y": v[3], "w": v[4], "h": v[5], "s": s})
    return out


def cover(g, masks, ellipse=False):
    """Phần diện tích box g bị hợp các vùng che phủ (0..1)."""
    if not masks or g["w"] <= 0 or g["h"] <= 0:
        return 0.0
    if ellipse:
        n, N = 0, 24
        for i in range(N):
            for j in range(N):
                px, py = g["x"] + (i + .5) / N * g["w"], g["y"] + (j + .5) / N * g["h"]
                for b in masks:
                    u, v = (px - b["x"] - b["w"] / 2) / (b["w"] / 2), (py - b["y"] - b["h"] / 2) / (b["h"] / 2)
                    if u * u + v * v <= 1:
                        n += 1
                        break
        return n / (N * N)
    R = []
    for b in masks:
        x0, y0 = max(g["x"], b["x"]), max(g["y"], b["y"])
        x1, y1 = min(g["x"] + g["w"], b["x"] + b["w"]), min(g["y"] + g["h"], b["y"] + b["h"])
        if x1 > x0 and y1 > y0:
            R.append((x0, y0, x1, y1))
    if not R:
        return 0.0
    A = g["w"] * g["h"]
    if len(R) == 1:
        return (R[0][2] - R[0][0]) * (R[0][3] - R[0][1]) / A
    xs, ys = sorted({v for r in R for v in (r[0], r[2])}), sorted({v for r in R for v in (r[1], r[3])})
    s = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(r[0] < cx < r[2] and r[1] < cy < r[3] for r in R):
                s += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return min(1.0, s / A)


def size_name(s):
    return next(n for lo, hi, n in SIZES if lo <= s < hi)


def evaluate(meta, gt, pred, a):
    targets = set(a.targets if a.targets is not None else meta.get("target_ids", []))
    n_frames = int(meta.get("frame_count") or (max(list(gt) + [0]) + 1))
    fps = float(meta.get("fps") or 0)
    pad = a.pad / 100
    c = dict(other=0, leak=0, tgt=0, tgt_blur=0, ign=0, head=0, head_open=0, masks=0, extra=0, f_other=0, f_leak=0)
    sum_c, by, persons, leaks, spans = 0.0, {}, {}, [], {}

    def bk(grp, name, leak):
        o = by.setdefault((grp, name), [0, 0])
        o[0] += 1
        o[1] += leak

    for f in range(n_frames):
        G = gt.get(f, [])
        M = [m for m in pred.get(f, []) if m["s"] >= a.thr]
        if pad:
            M = [{"x": m["x"] - m["w"] * pad, "y": m["y"] - m["h"] * pad, "w": m["w"] * (1 + 2 * pad), "h": m["h"] * (1 + 2 * pad)} for m in M]
        c["masks"] += len(M)
        any_leak = any_other = False
        for g in G:
            if g["cls"] == a.face:
                spans.setdefault(g["track"], []).append(f)
            if g["cls"] == a.head:
                if g["person"] in targets or g["ignore"]:
                    continue
                c["head"] += 1
                c["head_open"] += cover(g, M, a.ellipse) < a.cover
                continue
            if g["cls"] != a.face:
                continue
            if g["ignore"]:
                c["ign"] += 1
                continue
            cv = cover(g, M, a.ellipse)
            if g["person"] in targets:
                c["tgt"] += 1
                c["tgt_blur"] += cv >= a.tcover
                continue
            leak = cv < a.cover
            c["other"] += 1
            any_other = True
            sum_c += cv
            bk("Cỡ mặt (cạnh ngắn)", size_name(min(g["w"], g["h"])), leak)
            for k, (nm, vals) in ENUMS.items():
                bk(nm, vals[g[k]] if 0 <= g[k] < len(vals) else "?", leak)
            bk("Nhận ra được", IDENT_VI[g["ident"]], leak)
            bk("Gián tiếp", "có" if g["indirect"] else "không", leak)
            bk("Cắt ở mép", "có" if g["truncated"] else "không", leak)
            p = persons.setdefault(g["person"], dict(person=g["person"], n=0, leak=0, events=0, longest=0))
            p["n"] += 1
            if leak:
                c["leak"] += 1
                any_leak = True
                p["leak"] += 1
                leaks.append(dict(frame=f, person=g["person"], track=g["track"], cover=cv, w=g["w"], h=g["h"]))
        c["f_other"] += any_other
        c["f_leak"] += any_leak
        cand = [g for g in G if g["cls"] in (a.face, a.head)]
        for m in M:
            A = m["w"] * m["h"]
            hit = False
            for g in cand:
                iw = min(g["x"] + g["w"], m["x"] + m["w"]) - max(g["x"], m["x"])
                ih = min(g["y"] + g["h"], m["y"] + m["h"]) - max(g["y"], m["y"])
                if iw > 0 and ih > 0 and iw * ih >= 0.1 * min(A, g["w"] * g["h"]):
                    hit = True
                    break
            c["extra"] += not hit

    # gộp frame lộ liên tiếp của cùng track thành đoạn (cho phép hở 1 frame)
    leaks.sort(key=lambda q: (str(q["track"]), q["frame"]))
    events = []
    for q in leaks:
        e = events[-1] if events else None
        if e and e["track"] == q["track"] and q["frame"] - e["end"] <= 2:
            e["end"] = q["frame"]
            e["frames"] += 1
            e["min_cover"] = min(e["min_cover"], q["cover"])
        else:
            events.append(dict(track=q["track"], person=q["person"], start=q["frame"], end=q["frame"], frames=1, min_cover=q["cover"]))
    for e in events:
        fs = spans.get(e["track"], [])
        lo = hi = e["start"]  # lần xuất hiện liên tục chứa đầu đoạn
        S = set(fs)
        while lo - 1 in S:
            lo -= 1
        while hi + 1 in S:
            hi += 1
        head, tail = e["start"] - lo <= 2 and lo > 0, hi - e["end"] <= 2 and hi + 1 < n_frames
        e["position"] = "cả lần xuất hiện" if head and tail else "lúc vào khung" if head else "lúc ra khung" if tail else "giữa track"
        p = persons[e["person"]]
        p["events"] += 1
        p["longest"] = max(p["longest"], e["frames"])
    events.sort(key=lambda e: (e["start"], str(e["track"])))
    longest = max([e["frames"] for e in events] + [0])
    return dict(
        video=meta.get("video", ""), frames=n_frames, target_ids=sorted(targets),
        other_face_frames=c["other"], leak_face_frames=c["leak"], leak_rate=c["leak"] / max(c["other"], 1),
        mean_cover=sum_c / max(c["other"], 1), frames_with_others=c["f_other"], frames_with_leak=c["f_leak"],
        leak_events=len(events), longest_leak_frames=longest, longest_leak_seconds=longest / fps if fps else None,
        clean=c["leak"] == 0, target_face_frames=c["tgt"], target_blurred=c["tgt_blur"], ignored_face_frames=c["ign"],
        head_frames=c["head"], head_uncovered=c["head_open"], masks=c["masks"], masks_hitting_nothing=c["extra"],
        by_attribute=[dict(group=g, value=v, n=n, leak=l) for (g, v), (n, l) in by.items()],
        persons=sorted(persons.values(), key=lambda p: (-p["leak"], str(p["person"]))), events=events, leaks=leaks)


def stem(p):
    return SUFFIX.sub("", Path(p).stem).lower()


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt"), ap.add_argument("--pred")
    ap.add_argument("--gt-dir"), ap.add_argument("--pred-dir")
    ap.add_argument("--out", default="privacy_eval", help="thư mục ghi kết quả")
    ap.add_argument("--cover", type=float, default=0.9, help="độ phủ tối thiểu để coi là đã che (0–1)")
    ap.add_argument("--tcover", type=float, default=0.5, help="người được chọn bị coi là che nhầm khi độ phủ >= mức này")
    ap.add_argument("--thr", type=float, default=0.0, help="chỉ lấy vùng che có score >= mức này")
    ap.add_argument("--pad", type=float, default=0.0, help="nới mỗi cạnh box kết quả thêm N%% trước khi tính")
    ap.add_argument("--ellipse", action="store_true", help="vùng che là hình elip nội tiếp box")
    ap.add_argument("--face", default="face", help="tên nhãn mặt trong GT"), ap.add_argument("--head", default="head", help="tên nhãn đầu trong GT")
    ap.add_argument("--targets", type=lambda s: [int(x) for x in s.split(",") if x.strip()], help="ghi đè người được chọn, ví dụ 3,5 (để rỗng '' = che tất cả)")
    ap.add_argument("--pred-fmt", choices=["xywh", "xyxy", "cxcywh"], default="xywh")
    ap.add_argument("--pred-offset", type=int, default=0, help="số frame trong file kết quả ứng với frame đầu video (JSONL)")
    ap.add_argument("--pred-normalized", action="store_true", help="toạ độ kết quả chuẩn hoá 0–1")
    ap.add_argument("--max-leak-rate", type=float, help="thoát với mã 1 nếu tỷ lệ lộ gộp vượt mức này (dùng cho CI)")
    a = ap.parse_args()

    if a.gt and a.pred:
        pairs = [(Path(a.gt), Path(a.pred))]
    elif a.gt_dir and a.pred_dir:
        preds = {stem(p): p for p in sorted(Path(a.pred_dir).rglob("*")) if p.suffix.lower() in (".jsonl", ".json", ".txt")}
        pairs = [(g, preds.get(stem(g))) for g in sorted(Path(a.gt_dir).rglob("*.jsonl"))]
        for g, p in pairs:
            if p is None:
                print(f"! không có file kết quả cho {g.name}: coi như không che gì", file=sys.stderr)
    else:
        ap.error("cần --gt và --pred, hoặc --gt-dir và --pred-dir")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for g, p in pairs:
        meta, gt = load_gt(g)
        pred = load_pred(p, a.pred_fmt, a.pred_offset, meta.get("width"), meta.get("height"), a.pred_normalized) if p else {}
        r = evaluate(meta, gt, pred, a)
        r["gt_file"], r["pred_file"] = str(g), str(p) if p else None
        results.append(r)

    cols = ["video", "frames", "other_face_frames", "leak_face_frames", "leak_rate", "leak_events", "longest_leak_frames",
            "frames_with_leak", "target_face_frames", "target_blurred", "ignored_face_frames", "masks_hitting_nothing", "clean"]
    with open(out / "privacy_summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            w.writerow([r["video"] or Path(r["gt_file"]).name] + [round(r[k], 6) if isinstance(r[k], float) else r[k] for k in cols[1:]])
    with open(out / "privacy_events.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "person", "track", "start", "end", "frames", "min_cover", "position"])
        for r in results:
            for e in r["events"]:
                w.writerow([r["video"], e["person"], e["track"], e["start"], e["end"], e["frames"], round(e["min_cover"], 4), e["position"]])
    tot_o, tot_l = sum(r["other_face_frames"] for r in results), sum(r["leak_face_frames"] for r in results)
    tot_t, tot_tb = sum(r["target_face_frames"] for r in results), sum(r["target_blurred"] for r in results)
    agg = {}
    for r in results:
        for b in r["by_attribute"]:
            o = agg.setdefault((b["group"], b["value"]), [0, 0])
            o[0] += b["n"]
            o[1] += b["leak"]
    total = dict(videos=len(results), clean_videos=sum(r["clean"] for r in results), other_face_frames=tot_o, leak_face_frames=tot_l,
                 leak_rate=tot_l / max(tot_o, 1), leak_events=sum(r["leak_events"] for r in results),
                 longest_leak_frames=max([r["longest_leak_frames"] for r in results] + [0]),
                 target_face_frames=tot_t, target_blurred=tot_tb, target_blur_rate=tot_tb / max(tot_t, 1),
                 by_attribute=[dict(group=g, value=v, n=n, leak=l, leak_rate=l / max(n, 1)) for (g, v), (n, l) in agg.items()],
                 options=dict(cover=a.cover, tcover=a.tcover, thr=a.thr, pad=a.pad, ellipse=a.ellipse))
    for r in results:
        r.pop("leaks")
    (out / "privacy_report.json").write_text(json.dumps(dict(total=total, videos=results), ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'video':32} {'cần che':>8} {'lộ':>6} {'tỷ lệ lộ':>9} {'đoạn':>5} {'dài nhất':>8} {'★ che nhầm':>11}")
    for r in results:
        nm = (r["video"] or Path(r["gt_file"]).name)[:32]
        print(f"{nm:32} {r['other_face_frames']:8d} {r['leak_face_frames']:6d} {pct(r['leak_rate']):>9} {r['leak_events']:5d} {r['longest_leak_frames']:8d} {r['target_blurred']:5d}/{r['target_face_frames']:<5d}")
    print(f"\nTổng: {total['clean_videos']}/{total['videos']} video sạch · lộ {tot_l}/{tot_o} mặt-frame = {pct(total['leak_rate'])} · "
          f"{total['leak_events']} đoạn, dài nhất {total['longest_leak_frames']} frame · ★ che nhầm {pct(total['target_blur_rate'])}")
    print(f"Đã ghi {out}/privacy_summary.csv, privacy_events.csv, privacy_report.json")
    if a.max_leak_rate is not None and total["leak_rate"] > a.max_leak_rate:
        sys.exit(1)


if __name__ == "__main__":
    main()
