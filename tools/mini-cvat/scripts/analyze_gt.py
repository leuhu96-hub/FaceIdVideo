"""
analyze_gt.py — Phân tích tập ground truth (GT) cho bài toán face detection + tracking trong video.

Input: một hoặc nhiều file GT, mỗi file = 1 video. Hỗ trợ:
  - XML của Music Video dataset (<Video><Trajectory obj_id><Frame frame_no x y width height observation/>)
  - MOT (CVAT export "MOT 1.1"):
    frame,id,x,y,w,h,not_ignored,class_id,visibility        (frame bắt đầu từ 1)
  - Cột 7 (not_ignored) = 0  -> box bị bỏ qua khi đánh giá (đếm riêng)
  - Cột 9 (visibility) trong [0,1] nếu có -> phân tích mức độ bị che

Thông tin video (độ phân giải, FPS) lấy theo thứ tự ưu tiên:
  --videos (đọc trực tiếp file video) > seqinfo.ini cạnh thư mục gt (cấu trúc MOTChallenge) > --width/--height/--fps

Cách dùng:
  python analyze_gt.py gt_video1.txt gt_video2.txt --videos v1.mp4 v2.mp4 --out gt_report
  python analyze_gt.py data/*/gt/gt.txt --width 1920 --height 1080 --fps 30 --out gt_report
  # đủ 3 nhóm (A: chỉ GT, B: + video, C: + model)
  python analyze_gt.py gt1.txt --videos v1.mp4 --model face_detection_yunet_2026may.onnx \
         --rec face_recognition_sface_2021dec.onnx --out gt_report

Output (trong --out):
  report.html            báo cáo có biểu đồ, mở bằng trình duyệt
  suggested_params.json  tham số gợi ý cho eval_face_postproc.py
  issues.csv             danh sách nghi vấn lỗi nhãn (để mở lại trong CVAT kiểm tra)
  events.csv             các đoạn 2 người chồng nhau / đi ngang nhau (dễ đổi ID)
  split.csv              gợi ý chia video thành tập tune / test

Các phần phân tích:
  1. Tổng quan            5. Chuyển động (IoU giữa frame, tốc độ, dấu hiệu camera lia)
  2. Kích thước mặt       6. Tương tác giữa các ID (chồng nhau, đi ngang nhau)
  3. Hình dạng & vị trí   7. Chất lượng nhãn
  4. Thời gian & track    8. Gợi ý chia tập tune / test
"""
import argparse
import base64
import configparser
import csv
import io
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

YUNET_MIN, YUNET_MAX = 10, 300  # YuNet được huấn luyện cho mặt khoảng 10..300 px
SIZE_EDGES = [0, 10, 24, 64, 300, 1e9]
SIZE_NAMES = ["<10", "10-24", "24-64", "64-300", ">300"]
COLORS = ["#1D9E75", "#7F77DD", "#D85A30", "#BA7517", "#888780"]



def read_trajectory_xml(path):
    """
    Đọc GT dạng XML của Music Video dataset (Zhang et al., ECCV 2016):
      <Video start_frame=.. end_frame=..><Trajectory obj_id=..><Frame frame_no x y width height observation/>...
    Trả về list dòng MOT: [frame(1-based), id, x, y, w, h, not_ignored, class, visibility].
    Frame không có trong Trajectory = người đó không xuất hiện ở frame đó.
    observation = 0 được coi là box ignore (cột 7 = 0).
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    vid = root if root.tag == "Video" else root.find(".//Video")
    if vid is None:
        raise ValueError(f"{path}: không tìm thấy thẻ <Video>")
    start = int(vid.get("start_frame", 1))
    rows = []
    for tr in vid.iter("Trajectory"):
        tid = int(float(tr.get("obj_id")))
        for fr in tr.iter("Frame"):
            obs = fr.get("observation")
            rows.append([int(fr.get("frame_no")) - start + 1, tid, float(fr.get("x")), float(fr.get("y")),
                         float(fr.get("width")), float(fr.get("height")),
                         0.0 if obs is not None and float(obs) == 0 else 1.0, 1.0, -1.0])
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def is_xml(path):
    with open(path, "rb") as f:
        return f.read(256).lstrip().startswith(b"<")


# =========================================================== đọc dữ liệu
class Video:
    def __init__(self, name, path):
        self.name, self.path = name, path
        self.W = self.H = self.fps = None
        rows, self.bad = [], []
        if is_xml(path):
            rows = [[float(x) for x in r] for r in read_trajectory_xml(path)]
        with open(path) as f:
            for ln, r in enumerate(() if rows else csv.reader(f), 1):
                if not r or r[0].strip().startswith("#"):
                    continue
                try:
                    v = [float(x) for x in r[:9]] + [1.0] * max(0, 9 - len(r))
                    if len(r) < 6 or not np.all(np.isfinite(v[:6])):
                        raise ValueError
                except ValueError:
                    self.bad.append((ln, ",".join(r)[:60]))
                    continue
                if len(r) < 9:
                    v[8] = np.nan  # không có visibility
                if len(r) < 7:
                    v[6] = 1.0
                rows.append(v[:9])
        a = np.array(rows, float) if rows else np.zeros((0, 9))
        self.ign = a[a[:, 6] == 0]
        a = a[a[:, 6] != 0]
        self.fr, self.id = a[:, 0].astype(int), a[:, 1].astype(int)
        self.x, self.y, self.w, self.h = a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        self.vis = a[:, 8].copy()
        self.vis[self.vis == -1] = np.nan  # quy ước MOT: -1 = không dùng cột này
        self.n = len(a)
        self.has_vis = self.n > 0 and np.isfinite(self.vis).any()

    def tracks(self):
        """id -> chỉ số các box, đã sắp theo frame."""
        d = defaultdict(list)
        for i in np.argsort(self.fr, kind="stable"):
            d[self.id[i]].append(i)
        return {k: np.array(v) for k, v in d.items()}

    def by_frame(self):
        d = defaultdict(list)
        for i, f in enumerate(self.fr):
            d[f].append(i)
        return d


def load_meta(v, video_path, a):
    if video_path:
        import cv2

        cap = cv2.VideoCapture(video_path)
        v.W, v.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v.fps = cap.get(cv2.CAP_PROP_FPS) or None
        v.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
    else:
        ini = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(v.path))), "seqinfo.ini")
        if os.path.exists(ini):
            c = configparser.ConfigParser()
            c.read(ini)
            s = c["Sequence"]
            v.W, v.H = int(s.get("imWidth", 0)) or None, int(s.get("imHeight", 0)) or None
            v.fps = float(s.get("frameRate", 0)) or None
            v.nframes = int(s.get("seqLength", 0)) or None
        else:
            v.nframes = None
    v.W, v.H = v.W or a.width, v.H or a.height
    v.fps = v.fps or a.fps
    if not getattr(v, "nframes", None):
        v.nframes = int(v.fr.max()) if v.n else 0


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


# =========================================================== 1. tổng quan
def overview(vs):
    rows = []
    for v in vs:
        bf = v.by_frame()
        per = np.array([len(x) for x in bf.values()]) if bf else np.zeros(1)
        rows.append(dict(video=v.name, res=f"{v.W}x{v.H}" if v.W else "?", fps=v.fps or "?",
                         frames=v.nframes, labeled_frames=len(bf),
                         empty_frames=max(0, v.nframes - len(bf)), boxes=v.n, ids=len(set(v.id)),
                         faces_per_frame_mean=round(float(per.mean()), 2), faces_per_frame_max=int(per.max()),
                         ignored=len(v.ign)))
    return rows


# =========================================================== 2. kích thước
def sizes(vs):
    H = np.concatenate([v.h for v in vs])
    W = np.concatenate([v.w for v in vs])
    side = np.maximum(W, H)
    out = dict(h_p5=pct(H, 5), h_p50=pct(H, 50), h_p95=pct(H, 95))
    scale = {}
    for name, long_side in (("native", None), ("640", 640), ("960", 960), ("1280", 1280)):
        s = []
        for v in vs:
            k = 1.0 if (long_side is None or not v.W) else long_side / max(v.W, v.H)
            s.append(np.maximum(v.w, v.h) * k)
        s = np.concatenate(s)
        hist = np.histogram(s, SIZE_EDGES)[0] / max(len(s), 1)
        scale[name] = dict(zip(SIZE_NAMES, hist.round(4).tolist()))
        scale[name]["in_yunet_range"] = float(((s >= YUNET_MIN) & (s <= YUNET_MAX)).mean())
    out["by_input_size"] = scale
    best = max(scale, key=lambda k: scale[k]["in_yunet_range"])
    out["suggest_size"] = best
    out["side"] = side
    return out


# =========================================================== 3. hình dạng & vị trí
def shape_pos(vs):
    ar = np.concatenate([v.w / np.maximum(v.h, 1e-6) for v in vs])
    cx, cy, edge = [], [], 0
    nb = 0
    for v in vs:
        if not v.W:
            continue
        cx.append((v.x + v.w / 2) / v.W)
        cy.append((v.y + v.h / 2) / v.H)
        e = (v.x <= 2) | (v.y <= 2) | (v.x + v.w >= v.W - 2) | (v.y + v.h >= v.H - 2)
        edge += int(e.sum())
        nb += v.n
    lo, hi = pct(ar, 0.5), pct(ar, 99.5)
    return dict(ar=ar, ar_p=[pct(ar, q) for q in (0.5, 5, 50, 95, 99.5)],
                # nới 10% quanh phân vị 0,5..99,5, nhưng không hẹp hơn [0,6, 1,3]
                suggest_ar=[round(min(lo * 0.9, 0.6), 2), round(max(hi * 1.1, 1.3), 2)],
                cx=np.concatenate(cx) if cx else None, cy=np.concatenate(cy) if cy else None,
                edge_frac=edge / nb if nb else float("nan"))


# =========================================================== 4. thời gian
def temporal(vs):
    lens, gaps, segs, reapp = [], [], [], []
    per_id = []
    for v in vs:
        for tid, idx in v.tracks().items():
            f = v.fr[idx]
            span = f[-1] - f[0] + 1
            g = np.diff(f) - 1
            g = g[g > 0]
            lens.append(len(f))
            gaps += g.tolist()
            segs.append(len(g) + 1)
            fps = v.fps or 25
            long = g[g > fps]  # vắng mặt > 1 giây
            reapp += long.tolist()
            per_id.append(dict(video=v.name, id=int(tid), first=int(f[0]), last=int(f[-1]), boxes=len(f),
                               coverage=round(len(f) / span, 3), segments=len(g) + 1,
                               longest_gap=int(g.max()) if len(g) else 0))
    gaps = np.array(gaps)
    fps = np.median([v.fps for v in vs if v.fps]) if any(v.fps for v in vs) else 25
    short = gaps[gaps <= fps] if len(gaps) else gaps
    mid = gaps[gaps <= 2 * fps] if len(gaps) else gaps
    return dict(track_len=np.array(lens), gaps=gaps, segments=np.array(segs), per_id=per_id,
                n_reappear=len(reapp), reappear=np.array(reapp),
                # max_gap: phủ 95% khoảng trống <= 1 giây, tối thiểu 2 frame
                suggest_max_gap=int(np.clip(np.ceil(pct(short, 95)), 2, fps)) if len(short) else 8,
                # max_lost: phủ 99% khoảng trống <= 2 giây, trong khoảng [0,5 s, 2 s]
                suggest_max_lost=int(np.clip(np.ceil(pct(mid, 99)), fps / 2, 2 * fps)) if len(mid) else int(fps),
                fps=fps)


# =========================================================== 5. chuyển động
def motion(vs):
    ious, disp, scale = [], [], []
    cam_by = {}
    for v in vs:
        cam_pairs, cam_moving = 0, 0
        step = defaultdict(dict)  # frame -> id -> (dx, dy, h)
        for tid, idx in v.tracks().items():
            for a, b in zip(idx, idx[1:]):
                if v.fr[b] - v.fr[a] != 1:
                    continue
                A = (v.x[a], v.y[a], v.w[a], v.h[a])
                B = (v.x[b], v.y[b], v.w[b], v.h[b])
                ious.append(iou_xywh(A, B))
                dx = (B[0] + B[2] / 2) - (A[0] + A[2] / 2)
                dy = (B[1] + B[3] / 2) - (A[1] + A[3] / 2)
                h = max(A[3], 1)
                disp.append(np.hypot(dx, dy) / h)
                scale.append(B[3] / h)
                step[v.fr[b]][tid] = (dx / h, dy / h)
        for f, d in step.items():
            if len(d) < 2:
                continue
            m = np.array(list(d.values()))
            mag = np.linalg.norm(m, axis=1)
            if (mag > 0.08).all():
                u = m / mag[:, None]
                cos = u @ u.T
                cam_pairs += 1
                cam_moving += int((cos[np.triu_indices(len(m), 1)] > 0.9).all())
            else:
                cam_pairs += 1
        cam_by[v.name] = cam_moving / cam_pairs if cam_pairs else float("nan")
    ious, disp, scale = np.array(ious), np.array(disp), np.array(scale)
    return dict(iou=ious, disp=disp, scale=scale,
                iou_lt03=float((ious < 0.3).mean()) if len(ious) else float("nan"),
                iou_lt05=float((ious < 0.5).mean()) if len(ious) else float("nan"),
                disp_p95=pct(disp, 95),
                cam_by=cam_by,
                cam_frac=max([x for x in cam_by.values() if x == x], default=float("nan")),
                suggest_iou1=round(float(np.clip(pct(ious, 2), 0.1, 0.3)), 2) if len(ious) else 0.3)


# =========================================================== 6. tương tác
def interactions(vs, close=1.0):
    events = []
    for v in vs:
        bf = v.by_frame()
        state = {}  # (id1,id2) -> dict
        for f in sorted(bf):
            idx = bf[f]
            seen = set()
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    a, b = idx[i], idx[j]
                    ia, ib = sorted((v.id[a], v.id[b]))
                    if v.id[a] > v.id[b]:
                        a, b = b, a
                    A = (v.x[a], v.y[a], v.w[a], v.h[a])
                    B = (v.x[b], v.y[b], v.w[b], v.h[b])
                    o = iou_xywh(A, B)
                    d = np.hypot((A[0] + A[2] / 2) - (B[0] + B[2] / 2), (A[1] + A[3] / 2) - (B[1] + B[3] / 2))
                    if o > 0 or d < close * max(A[3], B[3]):
                        key = (ia, ib)
                        seen.add(key)
                        order = np.sign((A[0] + A[2] / 2) - (B[0] + B[2] / 2))
                        e = state.get(key)
                        if e is None or f - e["end"] > 1:
                            if e:
                                events.append(e)
                            e = dict(video=v.name, id1=ia, id2=ib, start=f, end=f, max_iou=0.0,
                                     first_order=order, crossed=False)
                            state[key] = e
                        e["end"] = f
                        e["max_iou"] = max(e["max_iou"], round(o, 3))
                        if order != 0 and e["first_order"] != 0 and order != e["first_order"]:
                            e["crossed"] = True
            for key in [k for k in state if k not in seen and state[k]["end"] < f - 1]:
                events.append(state.pop(key))
        events += list(state.values())
    for e in events:
        e["frames"] = e["end"] - e["start"] + 1
        e.pop("first_order", None)
    events.sort(key=lambda e: (-e["crossed"], -e["max_iou"]))
    return events


# =========================================================== 7. chất lượng nhãn
# =========================================================== 7b. nhóm A: nhất quán theo thời gian (chỉ cần GT)
def box_of(v, i):
    return (v.x[i], v.y[i], v.w[i], v.h[i])


def center(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])


def group_a(vs):
    """
    A1 quên gán nhãn cả đoạn       A2 xuất hiện / biến mất giữa khung   A3 box bỏ quên (đứng yên rồi nhảy)
    A4 hai ID đổi chỗ               A5 ID dùng lại cho người khác         A6 box rộng hẹp không ổn định
    A7 visibility / ignore bất thường
    """
    out = []

    def add(v, f, tid, kind, detail):
        out.append(dict(video=v.name, frame=int(f), id=int(tid), issue=kind, detail=detail))

    for v in vs:
        fps = v.fps or 25
        T = v.tracks()
        bf = v.by_frame()
        ids_at = {f: set(v.id[i] for i in idx) for f, idx in bf.items()}
        first, last = (min(bf), max(bf)) if bf else (0, 0)

        # ---- A1: frame bị thiếu người so với xung quanh (một ID đang "sống" nhưng vắng >= 3 frame liên tiếp,
        #          vị trí trước/sau gần nhau, không ở mép, không bị người khác che)
        for tid, idx in T.items():
            f = v.fr[idx]
            for k in np.where(np.diff(f) > 3)[0]:
                a, b = idx[k], idx[k + 1]
                gap = f[k + 1] - f[k] - 1
                if gap > 2 * fps:
                    continue  # vắng lâu: thường là ra khỏi khung / bị che thật -> để A5 xử lý
                A, B = box_of(v, a), box_of(v, b)
                h = max(A[3], B[3], 1)
                move = np.linalg.norm(center(A) - center(B)) / h
                near_edge = v.W and min(A[0], A[1], v.W - A[0] - A[2], v.H - A[1] - A[3]) < 0.5 * h
                occl = False
                for g in range(f[k] + 1, f[k + 1]):
                    for j in bf.get(g, []):
                        P = tuple((1 - (g - f[k]) / (gap + 1)) * np.array(A) + (g - f[k]) / (gap + 1) * np.array(B))
                        if iou_xywh(P, box_of(v, j)) > 0.1:
                            occl = True
                if move < 1.0 and not near_edge and not occl:
                    add(v, f[k] + 1, tid, "nghi quên nhãn cả đoạn",
                        f"vắng {gap} frame ({f[k] + 1}–{f[k + 1] - 1}) nhưng trước/sau gần như cùng chỗ, không ở mép, không ai che")

        # ---- A2: track bắt đầu / kết thúc giữa khung hình
        if v.W:
            for tid, idx in T.items():
                for i, when, lab in ((idx[0], v.fr[idx[0]], "xuất hiện"), (idx[-1], v.fr[idx[-1]], "biến mất")):
                    if when in (first, last) or len(idx) < 3:
                        continue
                    B = box_of(v, i)
                    m = min(B[0], B[1], v.W - B[0] - B[2], v.H - B[1] - B[3])
                    if m < max(B[3], 0.05 * v.H):
                        continue
                    near = [j for g in range(when - 3, when + 4) for j in bf.get(g, [])
                            if v.id[j] != tid and iou_xywh(B, box_of(v, j)) > 0]
                    if not near:
                        add(v, when, tid, f"{lab} giữa khung hình",
                            f"cách mép {m:.0f}px, không ai che: thiếu nhãn {'trước' if lab == 'xuất hiện' else 'sau'} frame này?")

        # ---- A3: box đứng yên tuyệt đối nhiều frame rồi nhảy -> quên cập nhật keyframe
        for tid, idx in T.items():
            run = 0
            for a, b in zip(idx, idx[1:]):
                A, B = box_of(v, a), box_of(v, b)
                if v.fr[b] - v.fr[a] != 1:
                    run = 0
                    continue
                d = np.abs(np.array(A) - np.array(B)).max()
                if d < 0.5:
                    run += 1
                else:
                    if run >= max(5, fps / 2) and np.linalg.norm(center(A) - center(B)) > 0.3 * max(A[3], 1):
                        add(v, v.fr[b], tid, "box bỏ quên",
                            f"đứng yên tuyệt đối {run} frame rồi nhảy: box có còn trên mặt trong đoạn đứng yên?")
                    run = 0

        # ---- A4: hai ID đổi chỗ cho nhau giữa 2 frame liên tiếp
        for f in sorted(bf):
            g = f + 1
            if g not in bf:
                continue
            cur = {v.id[i]: i for i in bf[f]}
            nxt = {v.id[i]: i for i in bf[g]}
            both = sorted(set(cur) & set(nxt))
            for x_ in range(len(both)):
                for y_ in range(x_ + 1, len(both)):
                    a, b = both[x_], both[y_]
                    Ca, Cb = center(box_of(v, cur[a])), center(box_of(v, cur[b]))
                    Na, Nb = center(box_of(v, nxt[a])), center(box_of(v, nxt[b]))
                    h = max(v.h[cur[a]], v.h[cur[b]], 1)
                    keep = np.linalg.norm(Ca - Na) + np.linalg.norm(Cb - Nb)
                    swap = np.linalg.norm(Ca - Nb) + np.linalg.norm(Cb - Na)
                    if swap < 0.5 * keep and keep > 0.6 * h:
                        add(v, g, a, "hai ID đổi chỗ", f"ID {a} và ID {b} tráo vị trí giữa frame {f} và {g}")

        # ---- A5: ID quay lại sau thời gian dài với kích thước / vị trí khác hẳn
        for tid, idx in T.items():
            f = v.fr[idx]
            for k in np.where(np.diff(f) > 5 * fps)[0]:
                A, B = box_of(v, idx[k]), box_of(v, idx[k + 1])
                r = B[3] / max(A[3], 1)
                d = np.linalg.norm(center(A) - center(B)) / max(A[3], B[3], 1)
                if r > 1.6 or r < 1 / 1.6 or d > 3:
                    add(v, f[k + 1], tid, "ID dùng lại?",
                        f"vắng {f[k + 1] - f[k] - 1} frame, quay lại với cỡ x{r:.2f}, lệch {d:.1f} lần chiều cao mặt: cùng một người?")

        # ---- A6: box rộng hẹp dao động mạnh (so với trung vị trượt 5 frame)
        for tid, idx in T.items():
            if len(idx) < 10:
                continue
            f, h = v.fr[idx], v.h[idx]
            ok = np.diff(f) == 1
            if ok.sum() < 5:
                continue
            jit = np.median(np.abs(np.diff(h))[ok] / np.maximum(h[:-1][ok], 1))
            if jit > 0.05:
                add(v, v.fr[idx[0]], tid, "box rộng hẹp không ổn định",
                    f"chiều cao box thay đổi trung vị {jit * 100:.0f}% mỗi frame (quy ước vẽ box chưa thống nhất?)")

        # ---- A7: visibility / ignore
        if v.has_vis:
            for i in np.where(np.isfinite(v.vis) & ((v.vis < 0) | (v.vis > 1)))[0]:
                add(v, v.fr[i], v.id[i], "visibility ngoài [0,1]", f"visibility={v.vis[i]}")
            for i in np.where(v.vis == 0)[0]:
                add(v, v.fr[i], v.id[i], "visibility = 0 nhưng không ignore", "mặt bị che hết vẫn bị tính khi chấm điểm")
        if len(v.ign) and v.n:
            frac = len(v.ign) / (len(v.ign) + v.n)
            if frac > 0.3:
                add(v, int(v.ign[0, 0]), -1, "quá nhiều box ignore", f"{frac * 100:.0f}% box bị ignore: đánh giá sẽ bỏ qua phần lớn dữ liệu")
    return out


def merge_issues(a, b):
    grp = defaultdict(list)
    for i in b:
        grp[(i["video"], i["id"], i["issue"])].append(i)
    out = list(a)
    for (vn, tid, kind), L in grp.items():
        if len(L) > 5:
            fs = sorted(x["frame"] for x in L)
            out.append(dict(video=vn, frame=fs[0], id=tid, issue=kind,
                            detail=f"{len(L)} lần, frame {fs[0]}–{fs[-1]}. Ví dụ: {L[0]['detail']}"))
        else:
            out += L
    out.sort(key=lambda x: (x["video"], x["frame"]))
    return out, None


def label_qa(vs):
    issues = []

    def add(v, f, tid, kind, detail):
        issues.append(dict(video=v.name, frame=int(f), id=int(tid), issue=kind, detail=detail))

    for v in vs:
        for ln, txt in v.bad:
            issues.append(dict(video=v.name, frame=-1, id=-1, issue="dòng dữ liệu hỏng",
                               detail=f"dòng {ln}: '{txt}' (thiếu cột / không phải số / NaN)"))
        uf = np.unique(v.fr)
        if len(uf) > 2:
            fps = v.fps or 25
            for k, f in enumerate(uf):
                prev = uf[k - 1] if k else None
                nxt = uf[k + 1] if k + 1 < len(uf) else None
                d = min(abs(f - x) for x in (prev, nxt) if x is not None)
                if d > 5 * fps:
                    add(v, f, -1, "frame tách biệt",
                        f"cách frame có nhãn gần nhất {d} frame (> 5 giây): sai số frame? lệch FPS?")
        for i in range(v.n):
            if v.fr[i] < 1 or (v.nframes and v.fr[i] > v.nframes):
                add(v, v.fr[i], v.id[i], "frame ngoài phạm vi video",
                    f"frame {v.fr[i]} (video có {v.nframes} frame; MOT đánh số từ 1 — có thể lệch 0/1)")
            if v.id[i] < 0:
                add(v, v.fr[i], v.id[i], "ID âm", "ID nên là số nguyên dương")
            if v.W and v.w[i] > 0 and v.h[i] > 0:
                inside = not (v.x[i] + v.w[i] <= 0 or v.y[i] + v.h[i] <= 0 or v.x[i] >= v.W or v.y[i] >= v.H)
                if inside and (v.x[i] < -1 or v.y[i] < -1 or v.x[i] + v.w[i] > v.W + 1 or v.y[i] + v.h[i] > v.H + 1):
                    vis_w = min(v.x[i] + v.w[i], v.W) - max(v.x[i], 0)
                    vis_h = min(v.y[i] + v.h[i], v.H) - max(v.y[i], 0)
                    add(v, v.fr[i], v.id[i], "box tràn ra ngoài khung",
                        f"chỉ {vis_w * vis_h / (v.w[i] * v.h[i]) * 100:.0f}% diện tích nằm trong ảnh (nên cắt box theo mép ảnh)")
                if v.h[i] > 0.6 * v.H or v.w[i] > 0.6 * v.W:
                    add(v, v.fr[i], v.id[i], "box quá lớn", f"{v.w[i]:.0f}x{v.h[i]:.0f} (> 60% khung hình)")
            if v.w[i] <= 0 or v.h[i] <= 0:
                add(v, v.fr[i], v.id[i], "kích thước <= 0", f"w={v.w[i]:.1f} h={v.h[i]:.1f}")
            elif max(v.w[i], v.h[i]) < 4:
                add(v, v.fr[i], v.id[i], "box quá nhỏ (<4px)", f"{v.w[i]:.1f}x{v.h[i]:.1f}")
            if v.W and (v.x[i] + v.w[i] < 0 or v.y[i] + v.h[i] < 0 or v.x[i] > v.W or v.y[i] > v.H):
                add(v, v.fr[i], v.id[i], "box nằm ngoài khung", f"x={v.x[i]:.0f} y={v.y[i]:.0f}")
            if v.h[i] > 0 and not (0.3 <= v.w[i] / v.h[i] <= 2.5):
                add(v, v.fr[i], v.id[i], "tỉ lệ w/h bất thường", f"w/h={v.w[i] / v.h[i]:.2f}")
        seen = defaultdict(int)
        for f, t in zip(v.fr, v.id):
            seen[(f, t)] += 1
        for (f, t), c in seen.items():
            if c > 1:
                add(v, f, t, "ID trùng trong 1 frame", f"{c} box")
        for f, idx in v.by_frame().items():
            for a_ in range(len(idx)):
                for b_ in range(a_ + 1, len(idx)):
                    a, b = idx[a_], idx[b_]
                    if v.id[a] != v.id[b]:
                        o = iou_xywh((v.x[a], v.y[a], v.w[a], v.h[a]), (v.x[b], v.y[b], v.w[b], v.h[b]))
                        if o > 0.7:
                            add(v, f, v.id[a], "1 mặt bị gán 2 ID", f"trùng với ID {v.id[b]} (IoU {o:.2f})")
        for tid, idx in v.tracks().items():
            if len(idx) <= 2:
                add(v, v.fr[idx[0]], tid, "track quá ngắn", f"ID chỉ có {len(idx)} box (gán nhầm ID? hoặc nên gộp vào ID khác)")
            for a, b in zip(idx, idx[1:]):
                if v.fr[b] - v.fr[a] != 1 or min(v.w[a], v.h[a], v.w[b], v.h[b]) <= 0:
                    continue
                h = max(v.h[a], 1)
                d = np.hypot(v.x[b] + v.w[b] / 2 - v.x[a] - v.w[a] / 2, v.y[b] + v.h[b] / 2 - v.y[a] - v.h[a] / 2)
                r = v.h[b] / h
                if d > 1.0 * h:
                    add(v, v.fr[b], tid, "nhảy vị trí", f"dịch {d / h:.1f} lần chiều cao mặt (có thể nhầm ID)")
                if r > 1.6 or r < 1 / 1.6:
                    add(v, v.fr[b], tid, "nhảy kích thước", f"x{r:.2f}")
            f = v.fr[idx]
            for k in np.where(np.diff(f) == 2)[0]:
                add(v, f[k] + 1, tid, "thiếu 1 frame giữa track", "có thể quên gán nhãn")
            # đoạn dài chuyển động thẳng đều tuyệt đối -> dấu hiệu nội suy keyframe của CVAT
            if len(idx) >= 12:
                cxs = v.x[idx] + v.w[idx] / 2
                d2 = np.abs(np.diff(cxs, 2)) + np.abs(np.diff(v.y[idx] + v.h[idx] / 2, 2)) + np.abs(np.diff(v.h[idx], 2))
                lin = d2 < 0.05
                run = 0
                for k, ok in enumerate(lin):
                    run = run + 1 if ok and f[k + 2] - f[k] == 2 else 0
                    if run == 10:
                        add(v, f[k + 2], tid, "đoạn nội suy (keyframe)",
                            "≥ 12 frame chuyển động thẳng đều tuyệt đối: kiểm tra box có còn ôm mặt")
    issues += group_a(vs)
    kinds = defaultdict(int)
    for i in issues:
        kinds[i["issue"]] += 1
    # cùng (video, ID, loại lỗi) lặp > 5 lần -> gộp thành 1 dòng cho dễ đọc (số đếm ở bảng tổng vẫn đầy đủ)
    grp = defaultdict(list)
    for i in issues:
        grp[(i["video"], i["id"], i["issue"])].append(i)
    out = []
    for (vn, tid, kind), L in grp.items():
        if len(L) > 5:
            fs = sorted(x["frame"] for x in L)
            out.append(dict(video=vn, frame=fs[0], id=tid, issue=kind,
                            detail=f"{len(L)} lần, frame {fs[0]}–{fs[-1]}. Ví dụ: {L[0]['detail']}"))
        else:
            out += L
    out.sort(key=lambda x: (x["video"], x["frame"]))
    return out, dict(kinds)


# =========================================================== 9. nhóm B: nội dung ảnh trong box (cần video)
def sharpness(gray64):
    """Độ nét = phương sai Laplacian trên crop 64x64 đã làm mịn nhẹ (giảm ảnh hưởng của nhiễu)."""
    import cv2

    return float(cv2.Laplacian(cv2.GaussianBlur(gray64, (3, 3), 0), cv2.CV_64F).var())


def video_checks(vs, a):
    """
    B1 tự gắn thuộc tính độ nét / độ sáng cho từng box (attrs.csv) + ảnh mẫu để chỉnh ngưỡng
    B2 box nằm trên vùng đồng màu (tường, vật che…)
    B3 frame lặp, cắt cảnh, ID đi qua cắt cảnh, số frame video khác khai báo
    """
    import heapq

    import cv2

    issues, attrs, samples = [], [], defaultdict(list)
    cuts_all = []
    rnd = np.random.default_rng(0)

    def add(v, f, tid, kind, detail):
        issues.append(dict(video=v.name, frame=int(f), id=int(tid), issue=kind, detail=detail))

    def keep(key, score, item, k=8):  # giữ k mẫu có score lớn nhất
        h = samples[key]
        if len(h) < k:
            heapq.heappush(h, (score, len(attrs), item))
        elif score > h[0][0]:
            heapq.heapreplace(h, (score, len(attrs), item))

    for v in vs:
        if not getattr(v, "video", None):
            continue
        cap = cv2.VideoCapture(v.video)
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bf = v.by_frame()
        prev = prev_h = None
        fr, dups, cuts = 0, [], []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            fr += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            sf = small.astype(np.float32)
            norm = (sf - sf.mean()) / (sf.std() + 1e-6)  # chuẩn hoá độ sáng: tắt/bật đèn không bị tính là cắt cảnh
            if prev is not None:
                d = float(np.abs(sf - prev).mean())
                ncc = float((norm * prev_h).mean())  # tương quan cấu trúc giữa 2 frame
                if d < a.dup_thr:
                    dups.append(fr)
                elif ncc < a.cut_ncc:
                    cuts.append(fr)
            prev, prev_h = sf, norm
            for i in bf.get(fr, []):
                x, y, w, h = v.x[i], v.y[i], v.w[i], v.h[i]
                x0, y0 = int(max(0, x)), int(max(0, y))
                x1, y1 = int(min(gray.shape[1], x + w)), int(min(gray.shape[0], y + h))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                c = cv2.resize(gray[y0:y1, x0:x1], (64, 64), interpolation=cv2.INTER_AREA)
                sh, br, ct = sharpness(c), float(c.mean()), float(c.std())
                bc = "mờ" if sh < a.blur_thr else ("nét" if sh > a.sharp_thr else "vừa")
                lc = "tối" if br < a.dark_thr else ("chói" if br > 200 else "bình thường")
                attrs.append(dict(video=v.name, frame=fr, id=int(v.id[i]), h=round(float(h), 1),
                                  sharpness=round(sh, 1), brightness=round(br, 1), contrast=round(ct, 1),
                                  blur=bc, light=lc))
                th = cv2.cvtColor(cv2.resize(img[y0:y1, x0:x1], (56, 56)), cv2.COLOR_BGR2RGB)
                label = f"{v.name[:10]}:{fr}:{int(v.id[i])}"
                keep("mờ nhất", -sh, (th, label, f"nét {sh:.0f}"))
                keep("nét nhất", sh, (th, label, f"nét {sh:.0f}"))
                keep("tối nhất", -br, (th, label, f"sáng {br:.0f}"))
                keep("ngẫu nhiên", float(rnd.random()), (th, label, f"nét {sh:.0f}"))
                if ct < a.flat_thr and br > a.dark_thr:
                    add(v, fr, v.id[i], "box trên vùng đồng màu",
                        f"độ tương phản {ct:.1f}: box có thể nằm trên tường / vật che, không phải mặt")
        cap.release()
        if declared and abs(declared - fr) > 1:
            add(v, fr, -1, "số frame video không khớp", f"file khai báo {declared} frame nhưng đọc được {fr}")
        if v.n and v.fr.max() > fr:
            add(v, int(v.fr.max()), -1, "GT dài hơn video", f"GT có nhãn tới frame {int(v.fr.max())}, video chỉ có {fr} frame")
        if dups:
            runs = np.split(np.array(dups), np.where(np.diff(dups) > 1)[0] + 1)
            add(v, dups[0], -1, "frame lặp",
                f"{len(dups)} frame giống hệt frame trước ({len(runs)} đoạn, ví dụ {runs[0][0]}–{runs[0][-1]}): video bị giật / ghép?")
        for c in cuts:
            cuts_all.append(dict(video=v.name, frame=c))
            before = {v.id[i] for i in bf.get(c - 1, [])}
            after = {v.id[i] for i in bf.get(c, [])}
            for tid in sorted(before & after):
                add(v, c, tid, "ID đi qua cắt cảnh",
                    "cảnh đổi đột ngột nhưng ID vẫn giữ: đúng là cùng người? (tracker thường mất dấu ở đây)")
    for k in samples:
        samples[k] = [x[2] for x in sorted(samples[k], key=lambda t: -t[0])]
    return dict(issues=issues, attrs=attrs, samples=dict(samples), cuts=cuts_all)


def charts_b(VB, a):
    out = {}
    if not VB["attrs"]:
        return out
    sh = np.array([r["sharpness"] for r in VB["attrs"]])
    br = np.array([r["brightness"] for r in VB["attrs"]])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].hist(np.maximum(sh, 1), bins=np.logspace(0, np.log10(max(sh.max(), 10) + 1), 40), color=COLORS[1])
    ax[0].set_xscale("log")
    for x, c in ((a.blur_thr, "#D85A30"), (a.sharp_thr, "#1D9E75")):
        ax[0].axvline(x, color=c, ls="--", lw=1)
    style(ax[0], "Độ nét box (đỏ: ngưỡng mờ, xanh: ngưỡng nét)", "phương sai Laplacian (log)", "số box")
    ax[1].hist(br, bins=40, range=(0, 255), color=COLORS[3])
    ax[1].axvline(a.dark_thr, color="#D85A30", ls="--", lw=1)
    ax[1].axvline(200, color="#D85A30", ls="--", lw=1)
    style(ax[1], "Độ sáng box (vạch: ngưỡng tối / chói)", "độ sáng trung bình", "số box")
    out["attr_hist"] = fig_b64(fig)
    rows = [k for k in ("mờ nhất", "ngẫu nhiên", "nét nhất", "tối nhất") if VB["samples"].get(k)]
    fig, axs = plt.subplots(len(rows), 8, figsize=(10, 1.55 * len(rows)))
    axs = np.atleast_2d(axs)
    for r, k in enumerate(rows):
        for c in range(8):
            ax = axs[r, c]
            ax.axis("off")
            if c < len(VB["samples"][k]):
                im, lab, val = VB["samples"][k][c]
                ax.imshow(im)
                ax.set_title(f"{val}\n{lab}", fontsize=6)
        axs[r, 0].text(-0.15, 0.5, k, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=9)
    out["samples"] = fig_b64(fig)
    return out


def section_b(VB, CB, a):
    if not VB["attrs"]:
        return [], ""
    at = VB["attrs"]
    n = len(at)
    blur = {k: sum(r["blur"] == k for r in at) / n for k in ("mờ", "vừa", "nét")}
    light = {k: sum(r["light"] == k for r in at) / n for k in ("tối", "bình thường", "chói")}
    f1 = lambda x: f"{x * 100:.1f}%"
    find = [f"Trong box GT: <b>{f1(blur['mờ'])}</b> mờ, <b>{f1(light['tối'])}</b> tối, {f1(light['chói'])} chói "
            f"(ngưỡng mờ {a.blur_thr}, tối {a.dark_thr}: xem ảnh mẫu ở phần 9 để chỉnh)."]
    if VB["cuts"]:
        find.append(f"{len(VB['cuts'])} lần cắt cảnh: tracker sẽ mất dấu ở các điểm này, cần gộp tracklet (HAC) để giữ ID.")
    by_v = defaultdict(lambda: defaultdict(int))
    for r in at:
        by_v[r["video"]]["box"] += 1
        by_v[r["video"]][r["blur"]] += 1
        by_v[r["video"]][r["light"]] += 1
    tab = [dict(video=k, box=d["box"], mờ=f1(d["mờ"] / d["box"]), nét=f1(d["nét"] / d["box"]),
                tối=f1(d["tối"] / d["box"]), chói=f1(d["chói"] / d["box"])) for k, d in by_v.items()]
    html = f"""<h2>9. Chất lượng ảnh trong box (cần video)</h2>
<p>Mỗi box GT được gắn thêm 2 thuộc tính, lưu ở <code>attrs.csv</code>: độ nét (mờ / vừa / nét) và độ sáng (tối / bình thường / chói).
Dùng file này để tính recall theo từng loại khi đánh giá model. Ngưỡng phụ thuộc camera và mức nén video:
nhìn ảnh mẫu bên dưới, nếu hàng "mờ nhất" vẫn rõ thì hạ <code>--blur-thr</code>, nếu hàng "ngẫu nhiên" có nhiều ảnh nhoè thì tăng lên.</p>
{img(CB.get('attr_hist'))}{table(tab)}<h3>Ảnh mẫu</h3>{img(CB.get('samples'))}
<h3>Cắt cảnh</h3>{table(VB['cuts'], limit=20)}"""
    return find, html


# =========================================================== 10. nhóm C: đối chiếu bằng model (cần --videos + --model)
def model_checks(vs, a):
    """
    C1 mặt thiếu nhãn trong GT      (model thấy mặt score cao nhiều frame liền, GT không có box)
    C2 box GT có thể không phải mặt (model không thấy mặt kể cả khi crop phóng to)
    C3 lệch quy ước vẽ box          (box GT so với box model: lệch tâm, to/nhỏ hơn có hệ thống)
    C4 cùng người nhưng 2 ID        (embedding mặt giống nhau, 2 ID không xuất hiện cùng lúc)  — cần --rec
    C5 một ID chứa 2 người          (embedding trong cùng 1 ID đổi đột ngột)                      — cần --rec
    """
    import sys

    import cv2

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from eval_face_postproc import YuNet, detect_in_roi, iou_matrix
    except ImportError:
        raise SystemExit("Nhóm C cần file eval_face_postproc.py đặt cùng thư mục với analyze_gt.py")
    det = YuNet(a.model, None if a.size is None else ("native" if a.size == "native" else int(a.size)))
    rec = cv2.FaceRecognizerSF.create(a.rec, "") if a.rec else None
    issues, missing, notface, bias, thumbs = [], [], [], [], {"missing": [], "notface": []}
    feats_by_track, pairs = {}, []

    def add(v, f, tid, kind, detail):
        issues.append(dict(video=v.name, frame=int(f), id=int(tid), issue=kind, detail=detail))

    def thumb(img, b, size=64):
        x, y, w, h = [int(t) for t in b[:4]]
        m = int(0.3 * max(w, h))
        H, W = img.shape[:2]
        c = img[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
        return cv2.cvtColor(cv2.resize(c, (size, size)), cv2.COLOR_BGR2RGB) if c.size else None

    for v in vs:
        if not getattr(v, "video", None):
            continue
        bf = v.by_frame()
        ign = defaultdict(list)
        for r in v.ign:
            ign[int(r[0])].append(r[2:6])
        cap = cv2.VideoCapture(v.video)
        fr = 0
        open_seg, gt_land = [], {}  # đoạn "mặt thiếu nhãn" đang mở; landmark model cho box GT
        tracks = v.tracks()
        while True:
            ok, img = cap.read()
            if not ok:
                break
            fr += 1
            if (fr - 1) % a.stride:
                continue
            D = det.detect(img, 0.3)
            gi = bf.get(fr, [])
            G = np.array([[v.x[i], v.y[i], v.w[i], v.h[i]] for i in gi]).reshape(-1, 4)
            Gi = np.array(ign.get(fr, [])).reshape(-1, 4)
            M = iou_matrix(D, G) if len(D) and len(G) else np.zeros((len(D), len(G)))
            # ---- C3 + landmark cho C4/C5: ghép GT với detection
            for j, i in enumerate(gi):
                if not len(D):
                    break
                k = int(np.argmax(M[:, j]))
                if M[k, j] >= 0.5 and D[k, 4] >= 0.6:
                    d = D[k]
                    bias.append(((v.x[i] + v.w[i] / 2 - d[0] - d[2] / 2) / d[2],
                                 (v.y[i] + v.h[i] / 2 - d[1] - d[3] / 2) / d[3], v.w[i] / d[2], v.h[i] / d[3], M[k, j]))
                    if d[4] >= a.emb_score and v.h[i] >= 40:
                        gt_land[i] = (fr, d)
            # ---- C2: box GT đủ lớn, thấy rõ, nhưng model không thấy mặt
            for j, i in enumerate(gi):
                vis_ok = not np.isfinite(v.vis[i]) or v.vis[i] >= 0.5
                if v.h[i] < a.min_face or not vis_ok or (len(D) and M[:, j].max() >= 0.3):
                    continue
                R = detect_in_roi(det, img, G[j], thr=0.3)
                if not len(R) or iou_matrix(G[j][None], R).max() < 0.3:
                    notface.append(dict(video=v.name, frame=fr, id=int(v.id[i])))
                    add(v, fr, v.id[i], "box GT không thấy mặt",
                        "model không tìm thấy mặt kể cả khi crop phóng to: vẽ nhầm vào nền / vật? (hoặc mặt quá nghiêng)")
                    if len(thumbs["notface"]) < 8:
                        thumbs["notface"].append((thumb(img, G[j]), f"{v.name[:10]}:{fr}:{int(v.id[i])}"))
            # ---- C1: model thấy mặt chắc chắn, GT không có box (kể cả vùng ignore)
            cand = []
            for k in range(len(D)):
                if D[k, 4] < a.miss_score:
                    continue
                if len(G) and M[k].max() >= 0.3:
                    continue
                if len(Gi) and iou_matrix(D[k][None], Gi).max() >= 0.3:
                    continue
                cand.append(D[k])
            still = []
            for seg in open_seg:
                hit = [c for c in cand if iou_matrix(seg["last"][None], c[None])[0, 0] >= 0.3]
                if hit:
                    seg["last"], seg["end"] = hit[0], fr
                    seg["n"] += 1
                    seg["score"] += hit[0][4]
                    cand = [c for c in cand if c is not hit[0]]
                    still.append(seg)
                elif fr - seg["end"] > 2 * a.stride:
                    missing.append(seg)
                else:
                    still.append(seg)
            for c in cand:
                still.append(dict(video=v.name, start=fr, end=fr, n=1, score=c[4], last=c, first=c,
                                  thumb=thumb(img, c)))
            open_seg = still
        missing += open_seg
        cap.release()

        # ---- C4/C5: embedding mặt theo track
        if rec is None:
            continue
        want = defaultdict(list)  # frame -> [(tid, det row)]
        for tid, idx in tracks.items():
            cand_i = [i for i in idx if i in gt_land]
            if not cand_i:
                continue
            pick = [cand_i[int(t)] for t in np.linspace(0, len(cand_i) - 1, min(2 * a.samples, len(cand_i)))]
            for i in pick:
                f, d = gt_land[i]
                want[f].append((int(tid), d))
        cap = cv2.VideoCapture(v.video)
        fr = 0
        F = defaultdict(list)
        while want:
            ok, img = cap.read()
            if not ok:
                break
            fr += 1
            for tid, d in want.pop(fr, []):
                face = np.r_[d[:4], d[5:15], d[4]].astype(np.float32)
                al = rec.alignCrop(img, face)
                if sharpness(cv2.resize(cv2.cvtColor(al, cv2.COLOR_BGR2GRAY), (64, 64))) < a.blur_thr:
                    continue  # mặt mờ: embedding không đáng tin
                e = rec.feature(al).ravel()
                F[tid].append((fr, e / np.linalg.norm(e), cv2.cvtColor(al, cv2.COLOR_BGR2RGB)))
        cap.release()
        spans = {int(t): (v.fr[idx].min(), v.fr[idx].max(), set(v.fr[idx].tolist())) for t, idx in tracks.items()}
        mean = {t: (lambda m: m / np.linalg.norm(m))(np.mean([x[1] for x in L], 0)) for t, L in F.items()}
        for t, L in F.items():
            feats_by_track[(v.name, t)] = L
            # C5: tìm điểm cắt mà trung bình embedding trước và sau khác nhau nhất (thay đổi kéo dài, không phải 1 frame xấu)
            L.sort(key=lambda x: x[0])
            best = None
            for k in range(2, len(L) - 1):
                m1 = np.mean([x[1] for x in L[:k]], 0)
                m2 = np.mean([x[1] for x in L[k:]], 0)
                c = float(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2)))
                if best is None or c < best[0]:
                    best = (c, k)
            if best and best[0] < a.diff_thr:
                c, k = best
                f1, f2 = L[k - 1][0], L[k][0]
                add(v, f2, t, "1 ID chứa 2 người?",
                    f"mặt trước frame {f1} và từ frame {f2} khác nhau rõ rệt (cosine {c:.2f} < {a.diff_thr}): đổi người giữa chừng?")
                pairs.append(dict(kind="1 ID 2 người", video=v.name, a=(t, f1), b=(t, f2), cos=c))
        ts = sorted(mean)
        for x_ in range(len(ts)):
            for y_ in range(x_ + 1, len(ts)):
                t1, t2 = ts[x_], ts[y_]
                if spans[t1][2] & spans[t2][2]:
                    continue  # xuất hiện cùng lúc -> chắc chắn 2 người (cannot-link)
                c = float(mean[t1] @ mean[t2])
                if c > a.same_thr:
                    later = t2 if spans[t2][0] > spans[t1][0] else t1
                    add(v, spans[later][0], later, "cùng người nhưng 2 ID",
                        f"ID {t1} và ID {t2} không xuất hiện cùng lúc, mặt giống nhau (cosine {c:.2f} > {a.same_thr})")
                    pairs.append(dict(kind="cùng người 2 ID", video=v.name, a=(t1, None), b=(t2, None), cos=c))
    seg_rows = []
    for m in missing:
        if m["n"] < a.min_len:
            continue
        b = m["first"]
        seg_rows.append(dict(video=m["video"], start=m["start"], end=m["end"], frames=m["n"] * a.stride,
                             x=round(float(b[0])), y=round(float(b[1])), w=round(float(b[2])), h=round(float(b[3])),
                             mean_score=round(float(m["score"] / m["n"]), 2)))
        issues.append(dict(video=m["video"], frame=m["start"], id=-1, issue="mặt thiếu nhãn trong GT",
                           detail=f"model thấy mặt (score {m['score'] / m['n']:.2f}) ở frame {m['start']}–{m['end']} "
                                  f"tại x={b[0]:.0f}, y={b[1]:.0f}, nhưng GT không có box"))
        if len(thumbs["missing"]) < 8 and m["thumb"] is not None:
            thumbs["missing"].append((m["thumb"], f"{m['video'][:10]}:{m['start']}"))
    return dict(issues=issues, missing=seg_rows, bias=np.array(bias).reshape(-1, 5), thumbs=thumbs,
                pairs=pairs, feats=feats_by_track, notface=notface)


def section_c(VC, a):
    f1 = lambda x: f"{x * 100:.1f}%"
    find, out = [], {}
    B = VC["bias"]
    html = "<h2>10. Đối chiếu bằng model</h2>"
    # ---- C3
    if len(B) >= 20:
        dx, dy, sw, sh, io = [float(np.median(B[:, k])) for k in range(5)]
        dev = max(abs(sw - 1), abs(sh - 1), abs(dx), abs(dy))
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        for k, (col, lab) in enumerate(((2, "w_GT / w_model"), (3, "h_GT / h_model"), (1, "lệch tâm dọc / h_model"))):
            ax[k].hist(B[:, col], bins=40, color=COLORS[k])
            ax[k].axvline(1 if col in (2, 3) else 0, color="#888780", ls="--", lw=1)
            style(ax[k], lab, None, "số box")
        out["bias"] = fig_b64(fig)
        verdict = ("<b>Có lệch quy ước</b>: box GT " + ("to" if (sw + sh) / 2 > 1 else "nhỏ") +
                   f" hơn box model khoảng {abs((sw + sh) / 2 - 1) * 100:.0f}%, tâm lệch dọc {dy * 100:+.0f}% chiều cao."
                   " IoU bị kéo xuống ở mọi box, không phải lỗi của model: hoặc thống nhất lại quy ước gán nhãn,"
                   " hoặc biến đổi box model theo các hệ số này trước khi chấm điểm.") if dev > 0.08 else             "Box GT và box model cùng quy ước (lệch &lt; 8%)."
        find.append(f"Quy ước box: IoU trung vị GT–model {io:.2f}. " + verdict.split(":")[0].replace("<b>", "").replace("</b>", "") +
                    ("." if dev <= 0.08 else ": xem phần 10."))
        html += f"""<h3>Quy ước vẽ box (GT so với model)</h3><p>{len(B)} cặp box GT–model khớp nhau. Trung vị: w_GT/w_model = {sw:.2f},
h_GT/h_model = {sh:.2f}, lệch tâm ngang {dx * 100:+.1f}%, dọc {dy * 100:+.1f}%, IoU {io:.2f}.<br>{verdict}</p>{img(out['bias'])}"""
    # ---- C1
    html += f"<h3>Mặt thiếu nhãn trong GT</h3><p>Model thấy mặt với score ≥ {a.miss_score} trong ≥ {a.min_len} frame liên tiếp mà GT không có box (đã trừ vùng ignore).</p>"
    html += strip_imgs(VC["thumbs"]["missing"]) + table(VC["missing"], limit=20)
    if VC["missing"]:
        find.append(f"<b>{len(VC['missing'])}</b> đoạn mặt có thể bị thiếu nhãn trong GT (model thấy rõ nhưng GT không có): xem phần 10.")
    # ---- C2
    html += f"<h3>Box GT không thấy mặt</h3><p>Box GT cao ≥ {a.min_face}px, visibility ≥ 0,5, nhưng model không tìm thấy mặt kể cả khi crop phóng to.</p>"
    html += strip_imgs(VC["thumbs"]["notface"]) + (f"<p>{len(VC['notface'])} box.</p>" if VC["notface"] else "<p class='m'>Không có.</p>")
    if VC["notface"]:
        find.append(f"{len(VC['notface'])} box GT mà model không thấy mặt: có thể vẽ nhầm vào nền.")
    # ---- C4/C5
    if a.rec:
        html += f"<h3>Kiểm tra danh tính (embedding mặt)</h3><p>Cùng người 2 ID: cosine &gt; {a.same_thr} và 2 ID không xuất hiện cùng lúc. 1 ID 2 người: hai mẫu liên tiếp có cosine &lt; {a.diff_thr}.</p>"
        html += pair_imgs(VC) + table([dict(loai=p["kind"], video=p["video"], ID_A=p["a"][0], ID_B=p["b"][0],
                                            frame=f"{p['a'][1]}→{p['b'][1]}" if p["a"][1] else "", cosine=round(p["cos"], 2)) for p in VC["pairs"]])
        n4 = sum(p["kind"] == "cùng người 2 ID" for p in VC["pairs"])
        n5 = len(VC["pairs"]) - n4
        if n4 or n5:
            find.append(f"Danh tính: {n4} cặp ID có thể là cùng một người, {n5} chỗ một ID có thể chứa 2 người.")
    else:
        html += "<p class='m'>Thêm <code>--rec face_recognition_sface_2021dec.onnx</code> để kiểm tra cùng người 2 ID / 1 ID 2 người.</p>"
    return find, html


def strip_imgs(items):
    if not items:
        return ""
    fig, axs = plt.subplots(1, 8, figsize=(10, 1.6))
    for c in range(8):
        axs[c].axis("off")
        if c < len(items) and items[c][0] is not None:
            axs[c].imshow(items[c][0])
            axs[c].set_title(items[c][1], fontsize=6)
    return img(fig_b64(fig))


def pair_imgs(VC):
    P = VC["pairs"][:6]
    if not P:
        return ""
    fig, axs = plt.subplots(len(P), 2, figsize=(3.2, 1.7 * len(P)))
    axs = np.atleast_2d(axs)
    for r, p in enumerate(P):
        for c, (tid, fr) in enumerate((p["a"], p["b"])):
            L = VC["feats"].get((p["video"], tid), [])
            pick = next((x for x in L if x[0] == fr), L[0] if L else None) if fr else (L[0] if L else None)
            axs[r, c].axis("off")
            if pick is not None:
                axs[r, c].imshow(pick[2])
                axs[r, c].set_title(f"ID {tid} · frame {pick[0]}", fontsize=7)
        axs[r, 0].text(-0.1, 0.5, f"{p['kind']}\ncos {p['cos']:.2f}", transform=axs[r, 0].transAxes,
                       ha="right", va="center", fontsize=7)
    return img(fig_b64(fig))


# =========================================================== 8. chia tập
def split(vs, sz, ev, test_frac=0.5):
    if len(vs) < 2:
        return [], "Chỉ có 1 video: nên chia theo đoạn thời gian dài (ví dụ nửa đầu / nửa sau) hoặc gán nhãn thêm video."
    ev_by = defaultdict(int)
    for e in ev:
        ev_by[e["video"]] += 1
    rows = []
    for v in vs:
        k = 640 / max(v.W, v.H) if v.W else 1
        small = float((np.maximum(v.w, v.h) * k < 24).mean()) if v.n else 0
        rows.append(dict(video=v.name, boxes=v.n, small_frac_at640=round(small, 3),
                         interactions=ev_by[v.name]))
    s = np.array([r["small_frac_at640"] for r in rows])
    it = np.array([r["interactions"] for r in rows], float)
    diff = s / (s.max() or 1) + it / (it.max() or 1)
    order = np.argsort(-diff)
    tot = sum(r["boxes"] for r in rows)
    acc = {"tune": 0, "test": 0}
    for rank, i in enumerate(order):
        want = "test" if acc["test"] / tot < test_frac * (acc["test"] + acc["tune"] + rows[i]["boxes"]) / tot else "tune"
        if rank < 2:
            want = ["test", "tune"][rank]
        rows[i]["split"] = want
        rows[i]["difficulty"] = round(float(diff[i]), 3)
        acc[want] += rows[i]["boxes"]
    return rows, f"tune: {acc['tune']} box, test: {acc['test']} box (cân bằng theo độ khó: tỉ lệ mặt nhỏ + số lần tương tác)"


# =========================================================== biểu đồ
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style(ax, title, xl=None, yl=None):
    ax.set_title(title, fontsize=11, loc="left")
    if xl:
        ax.set_xlabel(xl, fontsize=9)
    if yl:
        ax.set_ylabel(yl, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)


def charts(vs, S, SP, T, M):
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    h = SP["side"]
    bins = np.logspace(np.log10(max(h.min(), 2)), np.log10(h.max() + 1), 40)
    ax[0].hist(h, bins=bins, color=COLORS[0])
    ax[0].set_xscale("log")
    for x in (YUNET_MIN, 24, 64, YUNET_MAX):
        ax[0].axvline(x, color="#888780", ls="--", lw=0.8)
    style(ax[0], "Cỡ mặt ở độ phân giải gốc (cạnh dài, px)", "px (log)", "số box")
    names = list(SP["by_input_size"])
    bottom = np.zeros(len(names))
    for k, b in enumerate(SIZE_NAMES):
        vals = np.array([SP["by_input_size"][n][b] for n in names]) * 100
        ax[1].bar(names, vals, bottom=bottom, color=["#D85A30", "#BA7517", "#7F77DD", "#1D9E75", "#888780"][k], label=b)
        bottom += vals
    ax[1].legend(fontsize=7, title="cỡ mặt (px)", title_fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    style(ax[1], "Cỡ mặt sau khi đưa vào model", "cạnh dài input", "% box")
    out["size"] = fig_b64(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].hist(S["ar"], bins=40, color=COLORS[1])
    for x in S["suggest_ar"]:
        ax[0].axvline(x, color="#D85A30", ls="--", lw=1)
    style(ax[0], "Tỉ lệ w/h (vạch đỏ: ngưỡng gợi ý)", "w/h", "số box")
    if S["cx"] is not None:
        ax[1].hist2d(S["cx"], S["cy"], bins=24, range=[[0, 1], [0, 1]], cmap="Greens")
        ax[1].invert_yaxis()
        style(ax[1], "Vị trí tâm mặt trong khung hình", "x / width", "y / height")
    else:
        ax[1].text(0.5, 0.5, "Cần độ phân giải video", ha="center")
        ax[1].axis("off")
    out["shape"] = fig_b64(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].hist(T["track_len"], bins=30, color=COLORS[1])
    style(ax[0], "Độ dài track (số box / ID)", "box", "số ID")
    if len(T["gaps"]):
        ax[1].hist(T["gaps"], bins=np.arange(1, min(T["gaps"].max(), 200) + 2) if T["gaps"].max() < 200 else 50, color=COLORS[3])
        ax[1].axvline(T["suggest_max_gap"], color="#1D9E75", ls="--", lw=1, label="max_gap gợi ý")
        ax[1].axvline(T["suggest_max_lost"], color="#D85A30", ls="--", lw=1, label="max_lost gợi ý")
        ax[1].legend(fontsize=7)
    style(ax[1], "Khoảng trống bên trong 1 ID", "frame vắng mặt", "số lần")
    ax[2].hist(T["segments"], bins=np.arange(1, T["segments"].max() + 2) - 0.5, color=COLORS[0])
    style(ax[2], "Số đoạn của mỗi ID", "đoạn", "số ID")
    out["temporal"] = fig_b64(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    if len(M["iou"]):
        ax[0].hist(M["iou"], bins=40, range=(0, 1), color=COLORS[0])
        ax[0].axvline(0.3, color="#D85A30", ls="--", lw=1)
        ax[1].hist(np.clip(M["disp"], 0, 2), bins=40, color=COLORS[3])
    style(ax[0], "IoU cùng ID giữa 2 frame liên tiếp (vạch: 0,3)", "IoU", "số cặp")
    style(ax[1], "Độ dịch tâm / chiều cao mặt mỗi frame", "lần chiều cao mặt", "số cặp")
    out["motion"] = fig_b64(fig)

    if any(v.has_vis for v in vs):
        vis = np.concatenate([v.vis for v in vs])
        vis = vis[np.isfinite(vis)]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(vis, bins=20, range=(0, 1), color=COLORS[1])
        style(ax, "Visibility (1 = thấy rõ, 0 = bị che hết)", "visibility", "số box")
        out["vis"] = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.28 * min(len(T["per_id"]), 40)))
    rows = sorted(T["per_id"], key=lambda r: (r["video"], r["id"]))[:40]
    for k, r in enumerate(rows):
        v = next(x for x in vs if x.name == r["video"])
        f = np.sort(v.fr[v.id == r["id"]])
        br = np.split(f, np.where(np.diff(f) > 1)[0] + 1)
        ax.broken_barh([(b[0], b[-1] - b[0] + 1) for b in br], (k - 0.35, 0.7), color=COLORS[k % 4])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['video'][:12]}:{r['id']}" for r in rows], fontsize=7)
    style(ax, "Timeline từng ID (khoảng trắng = vắng mặt)" + (" — 40 ID đầu" if len(T["per_id"]) > 40 else ""), "frame")
    out["timeline"] = fig_b64(fig)
    return out


# =========================================================== báo cáo
def table(rows, cols=None, limit=None):
    if not rows:
        return "<p class='m'>Không có.</p>"
    cols = cols or list(rows[0])
    body = "".join("<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in cols) + "</tr>" for r in rows[:limit])
    more = f"<p class='m'>… và {len(rows) - limit} dòng nữa (xem file csv).</p>" if limit and len(rows) > limit else ""
    return f"<div class='t'><table><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr>{body}</table></div>{more}"


def img(b):
    return f"<img src='data:image/png;base64,{b}'/>" if b else ""


def report(path, vs, OV, SP, S, T, M, EV, QA, QAK, SPL, SPLmsg, P, CH, extra=()):
    f1 = lambda x: f"{x * 100:.1f}%" if x == x else "?"
    nb = sum(v.n for v in vs)
    sz = SP["by_input_size"]
    findings = []
    findings.append(f"Ở 640, <b>{f1(sz['640']['<10'])}</b> mặt nhỏ hơn 10 px (YuNet gần như không thấy); ở native là "
                    f"<b>{f1(sz['native']['<10'])}</b>. Cỡ input giữ nhiều mặt trong vùng 10–300 px nhất: <b>{SP['suggest_size']}</b>.")
    if sz["native"][">300"] > 0.01:
        findings.append(f"{f1(sz['native']['>300'])} mặt lớn hơn 300 px ở native: nên thu nhỏ ảnh hoặc chạy thêm 1 lần ở cỡ nhỏ.")
    findings.append(f"{f1(M['iou_lt03'])} cặp frame liên tiếp có IoU &lt; 0,3: "
                    + ("tracker chỉ dùng IoU sẽ hay mất dấu, cần Kalman tốt / CMC." if M["iou_lt03"] > 0.05 else "chuyển động chậm, tracker IoU + Kalman là đủ."))
    if M["cam_frac"] == M["cam_frac"]:
        lia = [k for k, x in M["cam_by"].items() if x == x and x > 0.2]
        findings.append("Camera lia / rung (mọi người cùng dịch một hướng): "
                        + (f"<b>{', '.join(lia)}</b> → nên bật CMC." if lia else "không thấy, camera khá ổn định."))
    findings.append(f"{T['n_reappear']} lần một ID vắng mặt hơn 1 giây rồi quay lại: "
                    + ("cần bước gộp tracklet dài hạn (HAC)." if T["n_reappear"] else "ít cần gộp dài hạn."))
    nc = sum(e["crossed"] for e in EV)
    findings.append(f"{len(EV)} đoạn 2 người ở sát nhau, trong đó {nc} đoạn đi ngang qua nhau: đây là nơi dễ ID switch nhất "
                    + ("→ nên có ReID." if nc else "."))
    for fx, _ in extra:
        findings += fx
    findings.append(f"{len(QA)} nghi vấn lỗi nhãn ({', '.join(f'{k}: {n}' for k, n in QAK.items()) or 'không có'}).")

    html = f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phân tích ground truth</title><style>
body{{font:15px/1.6 system-ui,sans-serif;max-width:1000px;margin:auto;padding:20px;color:#1f1e1d;background:#faf9f5}}
h1{{font-size:24px}}h2{{font-size:18px;margin-top:34px;border-bottom:1px solid #e3e1d8;padding-bottom:4px}}
img{{max-width:100%;background:#fff;border:1px solid #e3e1d8;border-radius:8px;margin:6px 0}}
.t{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;margin:8px 0}}td,th{{border:1px solid #e3e1d8;padding:4px 8px;text-align:left}}
th{{background:#f1efe8}}.m{{color:#6b6a65;font-size:13px}}.box{{background:#fff;border:1px solid #e3e1d8;border-radius:10px;padding:10px 16px}}
code{{background:#f1efe8;padding:1px 4px;border-radius:4px}}</style></head><body>
<h1>Phân tích ground truth</h1><p class="m">{len(vs)} video · {nb} box · {sum(len(set(v.id)) for v in vs)} ID</p>
<div class="box"><b>Kết luận nhanh</b><ul>{''.join(f'<li>{x}</li>' for x in findings)}</ul>
<b>Tham số gợi ý cho eval_face_postproc.py</b><br><code>{' '.join(f'--{k.replace("_", "-")} {v}' for k, v in P['cli'].items())}</code>
<p class="m">Tracker: max_lost={P['tracker']['max_lost']}, iou1={P['tracker']['iou1']} · Lọc hình học: ar={P['geometry']['ar']}</p></div>

<h2>1. Tổng quan</h2>{table(OV)}
<h2>2. Kích thước mặt</h2><p>Câu hỏi: chạy model ở độ phân giải nào? YuNet bắt tốt mặt khoảng 10–300 px.</p>{img(CH['size'])}
{table([dict(input=k, **{b: f1(v[b]) for b in SIZE_NAMES}, trong_vung_10_300=f1(v['in_yunet_range'])) for k, v in sz.items()])}
<h2>3. Hình dạng &amp; vị trí</h2><p>Tỉ lệ w/h (phân vị 0,5/5/50/95/99,5): {', '.join(f'{x:.2f}' for x in S['ar_p'])} → ngưỡng lọc gợi ý <b>{S['suggest_ar']}</b>.
Mặt chạm mép khung: <b>{f1(S['edge_frac'])}</b> (hay bị miss / box bị cắt).</p>{img(CH['shape'])}
{('<h3>Visibility</h3>' + img(CH['vis'])) if 'vis' in CH else '<p class="m">GT không có cột visibility. Nên gắn thuộc tính occlusion/blur trong CVAT để sau này tính recall theo từng loại.</p>'}
<h2>4. Thời gian &amp; track</h2><p>Độ dài track trung vị {pct(T['track_len'], 50):.0f} box. Khoảng trống bên trong ID: {len(T['gaps'])} lần,
trung vị {pct(T['gaps'], 50) if len(T['gaps']) else 0:.0f} frame. → <code>max_gap</code> ≈ {T['suggest_max_gap']} (phủ 95% khoảng trống ≤ 1 giây),
<code>max_lost</code> ≈ {T['suggest_max_lost']} (phủ 99% khoảng trống ≤ 2 giây). Vắng &gt; 1 giây rồi quay lại: {T['n_reappear']} lần.</p>
{img(CH['temporal'])}{img(CH['timeline'])}
<h2>5. Chuyển động</h2><p>IoU cùng ID giữa 2 frame: &lt;0,3: <b>{f1(M['iou_lt03'])}</b>, &lt;0,5: {f1(M['iou_lt05'])}. Độ dịch p95 = {M['disp_p95']:.2f} lần chiều cao mặt / frame.
Tỉ lệ frame có dấu hiệu camera lia (trong các frame có ≥ 2 người): {', '.join(f'{k}: {f1(x)}' for k, x in M['cam_by'].items())}.</p>{img(CH['motion'])}
<h2>6. Tương tác giữa các ID</h2><p>{len(EV)} đoạn có 2 người sát nhau (chồng box hoặc cách nhau &lt; 1 chiều cao mặt), {nc} đoạn đi ngang qua nhau.
Mở các đoạn này trong video để xem khi đánh giá tracking.</p>
{table(EV, ['video', 'id1', 'id2', 'start', 'end', 'frames', 'max_iou', 'crossed'], 15)}
<h2>7. Chất lượng nhãn</h2>{table([dict(loai=k, so_luong=n) for k, n in QAK.items()])}
{table(QA, ['video', 'frame', 'id', 'issue', 'detail'], 25)}
<h2>8. Chia tập tune / test</h2><p>{SPLmsg}</p>{table(SPL)}
{''.join(h for _, h in extra)}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt", nargs="+", help="file GT định dạng MOT (mỗi file = 1 video)")
    ap.add_argument("--videos", nargs="*", help="file video tương ứng (cùng thứ tự) để lấy độ phân giải/FPS")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--out", default="gt_report")
    ap.add_argument("--save-mot", action="store_true", help="ghi thêm GT đã đọc ra định dạng MOT (<tên>_mot.txt)")
    c = ap.add_argument_group("nhóm C (cần --videos + --model)")
    c.add_argument("--model", help="YuNet onnx (2023mar hoặc 2026may)")
    c.add_argument("--size", help='cỡ input cho YuNet: "native" (mặc định với 2026may), 640, 1280...')
    c.add_argument("--rec", help="SFace onnx (face_recognition_sface_2021dec.onnx) cho kiểm tra danh tính")
    c.add_argument("--miss-score", type=float, default=0.8, help="score tối thiểu để coi là mặt thiếu nhãn")
    c.add_argument("--min-len", type=int, default=3, help="số frame liên tiếp tối thiểu của đoạn thiếu nhãn")
    c.add_argument("--min-face", type=float, default=32, help="chỉ kiểm tra 'box không thấy mặt' với box cao >= ngưỡng")
    c.add_argument("--same-thr", type=float, default=0.363, help="cosine > ngưỡng -> cùng người (SFace: 0.363)")
    c.add_argument("--diff-thr", type=float, default=0.2, help="cosine < ngưỡng -> khác người")
    c.add_argument("--samples", type=int, default=8, help="số mặt lấy mẫu mỗi ID để tính embedding")
    c.add_argument("--emb-score", type=float, default=0.8, help="chỉ lấy embedding từ mặt có score model >= ngưỡng")
    c.add_argument("--stride", type=int, default=1, help="chạy model mỗi n frame (nhanh hơn, kém chi tiết hơn)")
    b = ap.add_argument_group("nhóm B (cần --videos)")
    b.add_argument("--blur-thr", type=float, default=60, help="độ nét < ngưỡng -> mờ")
    b.add_argument("--sharp-thr", type=float, default=150, help="độ nét > ngưỡng -> nét")
    b.add_argument("--dark-thr", type=float, default=50, help="độ sáng < ngưỡng -> tối")
    b.add_argument("--flat-thr", type=float, default=10, help="độ tương phản < ngưỡng -> vùng đồng màu")
    b.add_argument("--dup-thr", type=float, default=0.3, help="chênh lệch trung bình < ngưỡng -> frame lặp")
    b.add_argument("--cut-ncc", type=float, default=0.5, help="tương quan cấu trúc < ngưỡng -> cắt cảnh")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vs = []
    for k, p in enumerate(a.gt):
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "gt" and os.path.basename(os.path.dirname(os.path.abspath(p))) == "gt":  # MOTChallenge: <seq>/gt/gt.txt
            name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p))))
        v = Video(name, p)
        if a.save_mot:
            with open(os.path.join(a.out, name + "_mot.txt"), "w") as fo:
                for k in range(v.n):
                    vis = v.vis[k] if np.isfinite(v.vis[k]) else -1
                    fo.write(f"{v.fr[k]},{v.id[k]},{v.x[k]:.2f},{v.y[k]:.2f},{v.w[k]:.2f},{v.h[k]:.2f},1,1,{vis:g}\n")
        v.video = a.videos[k] if a.videos and k < len(a.videos) else None
        load_meta(v, v.video, a)
        vs.append(v)
    if not any(v.W for v in vs):
        print("Cảnh báo: không biết độ phân giải video -> bỏ qua phần vị trí và cỡ mặt theo input. Dùng --videos hoặc --width/--height.")

    OV = overview(vs)
    SP = sizes(vs)
    S = shape_pos(vs)
    T = temporal(vs)
    M = motion(vs)
    EV = interactions(vs)
    QA, QAK = label_qa(vs)
    extra = []
    VB = video_checks(vs, a) if any(v.video for v in vs) else None
    if VB:
        QA2, _ = merge_issues(QA, VB["issues"])
        QA = QA2
        for i in VB["issues"]:
            QAK[i["issue"]] = QAK.get(i["issue"], 0) + 1
        write_csv(os.path.join(a.out, "attrs.csv"), VB["attrs"])
        extra.append(section_b(VB, charts_b(VB, a), a))
    if a.model and any(v.video for v in vs):
        print("Nhóm C: chạy model trên video (có thể mất vài phút)...")
        VC = model_checks(vs, a)
        QA, _ = merge_issues(QA, VC["issues"])
        for i in VC["issues"]:
            QAK[i["issue"]] = QAK.get(i["issue"], 0) + 1
        write_csv(os.path.join(a.out, "missing_in_gt.csv"), VC["missing"])
        extra.append(section_c(VC, a))
    SPL, SPLmsg = split(vs, SP, EV)

    P = dict(cli=dict(size=SP["suggest_size"], max_gap=T["suggest_max_gap"]),
             tracker=dict(max_lost=T["suggest_max_lost"], iou1=M["suggest_iou1"]),
             geometry=dict(ar=S["suggest_ar"]),
             notes=dict(need_cmc=bool(M["cam_frac"] > 0.2) if M["cam_frac"] == M["cam_frac"] else None,
                        need_reid=bool(sum(e["crossed"] for e in EV) > 0),
                        need_hac=bool(T["n_reappear"] > 0)))
    with open(os.path.join(a.out, "suggested_params.json"), "w") as f:
        json.dump(P, f, indent=2, ensure_ascii=False)
    write_csv(os.path.join(a.out, "issues.csv"), QA)
    write_csv(os.path.join(a.out, "events.csv"), EV)
    write_csv(os.path.join(a.out, "split.csv"), SPL)
    write_csv(os.path.join(a.out, "ids.csv"), T["per_id"])
    CH = charts(vs, S, SP, T, M)
    report(os.path.join(a.out, "report.html"), vs, OV, SP, S, T, M, EV, QA, QAK, SPL, SPLmsg, P, CH, extra)

    print(f"{len(vs)} video, {sum(v.n for v in vs)} box, {sum(len(set(v.id)) for v in vs)} ID")
    print(f"Cỡ input gợi ý: {SP['suggest_size']}  | % mặt <10px ở 640: {SP['by_input_size']['640']['<10'] * 100:.1f}%")
    print(f"max_gap={T['suggest_max_gap']}  max_lost={T['suggest_max_lost']}  iou1={M['suggest_iou1']}  ar={S['suggest_ar']}")
    print(f"Tương tác: {len(EV)} đoạn ({sum(e['crossed'] for e in EV)} đi ngang)  | ID quay lại sau >1s: {T['n_reappear']}")
    print(f"Nghi vấn lỗi nhãn: {len(QA)}  {QAK}")
    print(f"-> {os.path.join(a.out, 'report.html')}")


if __name__ == "__main__":
    main()
