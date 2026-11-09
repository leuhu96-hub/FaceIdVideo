"""
eval_face_postproc.py  (v2)
So sánh face detection TRƯỚC và SAU từng bước post-processing trên video có ground truth.

Model ví dụ: YuNet 2023mar (OpenCV Zoo, MIT), chạy bằng onnxruntime để lấy được output THÔ trước NMS.
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Cài đặt:  pip install opencv-python onnxruntime numpy scipy

Ground truth: định dạng MOT (CVAT export "MOT 1.1"), mỗi dòng: frame,id,x,y,w,h,...  (frame bắt đầu từ 1)
              hoặc XML của Music Video dataset (<Video><Trajectory obj_id><Frame frame_no .../>), tự nhận dạng
Person box (tuỳ chọn, cho ràng buộc "mặt nằm trong vùng đầu người"): cùng định dạng MOT.

Model input động (YuNet 2026may) cho phép chọn độ phân giải: --size native | 640 | 1280 ...
  (2026may có weights giống hệt 2023mar; khác biệt duy nhất là chạy được ở độ phân giải bất kỳ)

Cách dùng:
  # 1) Chạy detector 1 lần, lưu toàn bộ ứng viên TRƯỚC NMS (score >= 0.2)
  python eval_face_postproc.py detect --video test.mp4 --model yunet.onnx --out raw.txt

  # 2) Đánh giá baseline + từng bước post-processing (cộng dồn)
  python eval_face_postproc.py eval --raw raw.txt --gt gt.txt --video test.mp4 --model yunet.onnx \
         [--person person.txt] [--out-dir results/]

Các biến thể (mỗi dòng = dòng trên + 1 bước), số trong ngoặc = số bước trong danh sách post-processing:
  0. baseline           : NMS thường (IoU 0.3), ngưỡng 0.6  ~ giống cv2.FaceDetectorYN mặc định
  1. + weighted NMS     (1)  box = trung bình có trọng số các box trùng   (--nms soft để thử Soft-NMS)
  2. + geometry/person  (3)  lọc tỉ lệ, kích thước, landmark; mặt phải nằm vùng đầu person box
  3. + verifier         (4)  box score lưng chừng -> crop, đưa mặt về ~96px, detect lại để xác nhận
  4. + tracker          (2,8) Kalman + Hungarian 2 vòng (kiểu ByteTrack): ngưỡng kép, xác nhận track
  5. + ROI re-detect    (5)  frame track bị miss -> crop vùng Kalman dự đoán, detect lại ngưỡng thấp
  6. + optical flow     (7)  khoảng trống còn lại -> kéo box theo optical flow (Lucas-Kanade)
  7. + interpolation    (6)  khoảng trống ngắn còn lại -> nội suy tuyến tính
  8. + RTS smoothing    (9)  làm mượt box bằng Kalman 2 chiều (xuôi + ngược)
"""
import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Một detection = np.array 15 phần tử: [x, y, w, h, score, lm_x0, lm_y0, ..., lm_x4, lm_y4]
# (thứ tự landmark YuNet: mắt phải, mắt trái, mũi, khoé miệng phải, khoé miệng trái; NaN nếu không có)
SIZE_BUCKETS = [(0, 24, "<24px"), (24, 64, "24-64px"), (64, 1e9, ">64px")]
IOU_MATCH = 0.5


# =========================================================== YuNet (onnxruntime, có output thô)
class YuNet:
    STRIDES = (8, 16, 32)

    def __init__(self, path, size=None):
        """
        size: cạnh dài (px) đưa vào model, hoặc "native" = giữ nguyên độ phân giải gốc.
          - Model input cố định (2023mar, 640x640): mặc định 640; "native" chỉ pad, không thu nhỏ nếu ảnh <= 640.
          - Model input động (2026may):             mặc định "native" (chỉ pad lên bội số 32).
        """
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        shp = self.sess.get_inputs()[0].shape
        self.dynamic = not all(isinstance(v, int) for v in shp[2:])
        self.fixed = None if self.dynamic else int(shp[2])
        self.size = size if size is not None else ("native" if self.dynamic else self.fixed)

    def raw(self, img, thr=0.2, size=None):
        """Toàn bộ ứng viên TRƯỚC NMS, toạ độ theo ảnh gốc. Trả về mảng Nx15."""
        H, W = img.shape[:2]
        size = self.size if size is None else size
        if size == "native":
            s = 1.0 if self.dynamic else min(1.0, self.fixed / max(H, W))
        else:
            s = float(size) / max(H, W)
            if not self.dynamic:
                s = min(s, self.fixed / max(H, W))
        nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
        if self.dynamic:
            ph, pw = -(-nh // 32) * 32, -(-nw // 32) * 32
        else:
            ph = pw = self.fixed
        pad = np.zeros((ph, pw, 3), np.uint8)
        pad[:nh, :nw] = img if (nh, nw) == (H, W) else cv2.resize(img, (nw, nh))
        blob = pad.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, không chuẩn hoá (giống OpenCV)
        out = dict(zip([o.name for o in self.sess.get_outputs()], self.sess.run(None, {"input": blob})))
        res = []
        for st in self.STRIDES:
            n = pw // st  # số cột của lưới anchor
            cls = np.clip(out[f"cls_{st}"].reshape(-1), 0, 1)
            obj = np.clip(out[f"obj_{st}"].reshape(-1), 0, 1)
            score = np.sqrt(cls * obj)
            keep = np.where(score >= thr)[0]
            if not len(keep):
                continue
            bb = out[f"bbox_{st}"].reshape(-1, 4)[keep]
            kp = out[f"kps_{st}"].reshape(-1, 10)[keep]
            r, c = (keep // n).astype(np.float32), (keep % n).astype(np.float32)
            cx, cy = (c + bb[:, 0]) * st, (r + bb[:, 1]) * st
            w, h = np.exp(bb[:, 2]) * st, np.exp(bb[:, 3]) * st
            lm = np.empty((len(keep), 10), np.float32)
            lm[:, 0::2] = (kp[:, 0::2] + c[:, None]) * st
            lm[:, 1::2] = (kp[:, 1::2] + r[:, None]) * st
            d = np.column_stack([cx - w / 2, cy - h / 2, w, h, score[keep], lm])
            d[:, [0, 1, 2, 3]] /= s
            d[:, 5:] /= s
            res.append(d)
        return np.concatenate(res) if res else np.zeros((0, 15), np.float32)

    def detect(self, img, thr=0.3, nms_iou=0.3, size=None):
        return hard_nms(self.raw(img, thr, size), nms_iou)


# =========================================================== IO

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


def load_mot(path, has_score):
    out = defaultdict(list)
    if not has_score and is_xml(path):  # GT dạng XML (Music Video dataset)
        for v in read_trajectory_xml(path):
            if v[6] == 0:
                continue
            d = np.full(15, np.nan, np.float32)
            d[:4] = v[2:6]
            d[4] = 1.0
            out[int(v[0])].append(d)
        return {k: np.array(v) for k, v in out.items()}
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            v = [float(x) for x in row]
            d = np.full(15, np.nan, np.float32)
            d[:4] = v[2:6]
            d[4] = v[6] if has_score else 1.0
            if has_score and len(v) >= 17:
                d[5:] = v[7:17]
            out[int(v[0])].append(d)
    return {k: np.array(v) for k, v in out.items()}


def save_mot(dets, path):
    with open(path, "w") as f:
        for fr in sorted(dets):
            for d in dets[fr]:
                f.write(",".join([str(fr), "-1"] + [f"{x:.2f}" for x in d[:5]]
                                 + [f"{x:.2f}" for x in d[5:]]) + "\n")


def iter_frames(video, wanted=None):
    """Đọc video tuần tự; chỉ yield các frame trong `wanted` (None = tất cả). Frame đánh số từ 1."""
    cap = cv2.VideoCapture(video)
    fr, last = 0, max(wanted) if wanted else None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        fr += 1
        if wanted is None or fr in wanted:
            yield fr, img
        if last is not None and fr >= last:
            break
    cap.release()


def parse_size(v):
    return None if v is None else ("native" if v == "native" else int(v))


def cmd_detect(a):
    det = YuNet(a.model, parse_size(a.size))
    print(f"Model: {a.model} | input {'động' if det.dynamic else 'cố định ' + str(det.fixed)} | size={det.size}")
    n = 0
    with open(a.out, "w") as f:
        for fr, img in iter_frames(a.video):
            for d in det.raw(img, 0.2):
                f.write(",".join([str(fr), "-1"] + [f"{x:.2f}" for x in d[:4]] + [f"{d[4]:.4f}"]
                                 + [f"{x:.2f}" for x in d[5:]]) + "\n")
            n = fr
    print(f"Đã ghi {a.out} ({n} frame, ứng viên trước NMS)")


# =========================================================== hình học
def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def iou_matrix(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    A, B = np.asarray(A)[:, :4], np.asarray(B)[:, :4]
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 0] + A[:, None, 2], B[None, :, 0] + B[None, :, 2])
    y2 = np.minimum(A[:, None, 1] + A[:, None, 3], B[None, :, 1] + B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None] - inter
    return inter / np.maximum(ua, 1e-9)


# =========================================================== bước 1: NMS
def hard_nms(D, thr=0.3):
    if len(D) == 0:
        return D
    D = D[np.argsort(-D[:, 4])]
    keep, alive = [], np.ones(len(D), bool)
    M = iou_matrix(D, D)
    for i in range(len(D)):
        if alive[i]:
            keep.append(i)
            alive &= M[i] < thr
    return D[keep]


def weighted_nms(D, thr=0.3):
    """Cụm các box trùng nhau; box đại diện = trung bình box + landmark theo trọng số score."""
    if len(D) == 0:
        return D
    D = D[np.argsort(-D[:, 4])]
    M = iou_matrix(D, D)
    alive, out = np.ones(len(D), bool), []
    for i in range(len(D)):
        if not alive[i]:
            continue
        grp = np.where(alive & (M[i] >= thr))[0]
        w = D[grp, 4:5]
        d = D[i].copy()
        d[:4] = (D[grp, :4] * w).sum(0) / w.sum()
        if not np.isnan(D[grp, 5:]).any():
            d[5:] = (D[grp, 5:] * w).sum(0) / w.sum()
        out.append(d)
        alive[grp] = False
    return np.array(out)


def soft_nms(D, sigma=0.5, min_score=0.2):
    """Soft-NMS (Gaussian): không xoá box trùng mà giảm score theo IoU."""
    D = D.copy()
    out = []
    while len(D):
        i = int(np.argmax(D[:, 4]))
        best = D[i]
        out.append(best)
        D = np.delete(D, i, 0)
        if len(D):
            D[:, 4] *= np.exp(-(iou_matrix(best[None], D)[0] ** 2) / sigma)
            D = D[D[:, 4] >= min_score]
    return np.array(out) if out else np.zeros((0, 15), np.float32)


# =========================================================== bước 3: hình học + person
def geometry_ok(d, min_size=8, ar=(0.5, 1.6)):
    x, y, w, h = d[:4]
    if min(w, h) < min_size or not (ar[0] <= w / h <= ar[1]):
        return False
    lm = d[5:].reshape(5, 2)
    if not np.isnan(lm).any():
        eyes_y, mouth_y = lm[:2, 1].mean(), lm[3:, 1].mean()
        inside = np.all((lm[:, 0] >= x - 0.15 * w) & (lm[:, 0] <= x + 1.15 * w)
                        & (lm[:, 1] >= y - 0.15 * h) & (lm[:, 1] <= y + 1.15 * h))
        eye_dist = np.hypot(*(lm[0] - lm[1]))
        if eyes_y >= mouth_y or not inside or not (0.1 * w <= eye_dist <= 0.9 * w):
            return False
    return True


def in_head_region(d, persons):
    """Tâm mặt nằm trong 40% phía trên của một person box."""
    cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
    for p in persons:
        if p[0] <= cx <= p[0] + p[2] and p[1] - 0.05 * p[3] <= cy <= p[1] + 0.4 * p[3]:
            return True
    return False


# =========================================================== crop + detect lại (dùng cho bước 4, 5)
def detect_in_roi(det, img, box, face_px=96, thr=0.3, expand=2.0):
    """
    Crop vùng quanh `box` (nới rộng `expand` lần), co giãn sao cho mặt ~face_px pixel
    (mặt nhỏ được phóng to, box to bị thu nhỏ -> detector nhìn ở tỉ lệ KHÁC lần đầu),
    chạy detector, trả về toạ độ gốc.
    """
    H, W = img.shape[:2]
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    side = expand * max(box[2], box[3], 8)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((0, 15), np.float32)
    crop = img[y0:y1, x0:x1]
    k = float(np.clip(face_px / max(box[2], box[3], 1), 0.25, 6.0))
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    D = det.detect(crop, thr, size="native")  # giữ cỡ mặt ~face_px, không để model co giãn lại
    if len(D):
        D[:, :4] /= k
        D[:, 5:] /= k
        D[:, [0, 5, 7, 9, 11, 13]] += x0
        D[:, [1, 6, 8, 10, 12, 14]] += y0
    return D


def verify(dets, video, det, low=0.3, high=0.6, boost_iou=0.4):
    """Bước 4: box có low <= score < high -> crop phóng to, detect lại; nếu thấy mặt thì nâng score."""
    need = {fr for fr, D in dets.items() if len(D) and np.any((D[:, 4] >= low) & (D[:, 4] < high))}
    out = {fr: D.copy() for fr, D in dets.items()}
    for fr, img in iter_frames(video, need):
        for d in out[fr]:
            if low <= d[4] < high:
                R = detect_in_roi(det, img, d, thr=low)
                if len(R):
                    o = iou_matrix(d[None], R)[0]
                    j = int(np.argmax(o))
                    if o[j] >= boost_iou:
                        d[4] = max(d[4], R[j, 4])
    return out


# =========================================================== bước 2, 8: Kalman tracker kiểu ByteTrack
class KF:
    """Kalman vận tốc không đổi trên [cx, cy, w, h]; nhiễu tỉ lệ với chiều cao box (như SORT/ByteTrack)."""
    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    Hm = np.eye(4, 8)

    def __init__(self, box, wp=1 / 20, wv=1 / 160):
        self.wp, self.wv = wp, wv
        h = box[3]
        self.x = np.r_[xywh2c(box), 0, 0, 0, 0].astype(float)
        self.P = np.diag(np.r_[[2 * wp * h] * 4, [10 * wv * h] * 4] ** 2)

    def Q(self):
        h = self.x[3]
        return np.diag(np.r_[[self.wp * h] * 4, [self.wv * h] * 4] ** 2)

    def R(self):
        return np.diag([(self.wp * self.x[3]) ** 2] * 4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q()
        self.x[2:4] = np.maximum(self.x[2:4], 2)
        return c2xywh(self.x[:4])

    def update(self, box):
        S = self.Hm @ self.P @ self.Hm.T + self.R()
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (xywh2c(box) - self.Hm @ self.x)
        self.P = (np.eye(8) - K @ self.Hm) @ self.P


def xywh2c(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]], float)


def c2xywh(c):
    return np.array([c[0] - c[2] / 2, c[1] - c[3] / 2, c[2], c[3]], float)


class Track:
    def __init__(self, tid, d, fr):
        self.id, self.kf = tid, KF(d[:4])
        self.boxes = {fr: d.copy()}      # frame -> detection (15 phần tử)
        self.src = {fr: "det"}           # nguồn: det / roi / flow / interp
        self.pred = {}                   # frame bị miss -> box Kalman dự đoán
        self.last = fr

    @property
    def score(self):
        return float(np.mean([self.boxes[f][4] for f, s in self.src.items() if s == "det"]))


def track(dets, high=0.6, low=0.3, iou1=0.3, iou2=0.5, max_lost=15, min_hits=3):
    """
    Vòng 1: box score >= high ghép với mọi track (Hungarian trên 1-IoU với box Kalman dự đoán).
    Vòng 2: box low <= score < high ghép với track còn lại (ngưỡng kép / hysteresis).
    Box high chưa ghép -> track mới. Track mất > max_lost frame -> kết thúc.
    Chỉ giữ track có >= min_hits detection (xác nhận).
    """
    active, done, nid = [], [], 0
    for fr in range(1, max(dets) + 1 if dets else 1):
        D = dets.get(fr, np.zeros((0, 15)))
        P = np.array([t.kf.predict() for t in active]) if active else np.zeros((0, 4))
        hi, lo = D[D[:, 4] >= high], D[(D[:, 4] >= low) & (D[:, 4] < high)]
        un_t = list(range(len(active)))
        for pool, thr in ((hi, iou1), (lo, iou2)):
            if not len(pool) or not un_t:
                if pool is hi:
                    used_hi = set()
                continue
            M = iou_matrix(P[un_t], pool)
            r, c = linear_sum_assignment(-M)
            matched_t, used = set(), set()
            for i, j in zip(r, c):
                if M[i, j] >= thr:
                    t = active[un_t[i]]
                    t.kf.update(pool[j][:4])
                    t.boxes[fr], t.src[fr], t.last = pool[j].copy(), "det", fr
                    matched_t.add(un_t[i])
                    used.add(j)
            un_t = [i for i in un_t if i not in matched_t]
            if pool is hi:
                used_hi = used
        for j in range(len(hi)):
            if j not in used_hi:
                active.append(Track(nid, hi[j], fr))
                nid += 1
        keep = []
        for i, t in enumerate(active):
            if i in un_t and t.last != fr:
                if fr - t.last > max_lost:
                    done.append(t)
                    continue
                t.pred[fr] = P[i] if i < len(P) else t.kf.x[:4]
            keep.append(t)
        active = keep
    return [t for t in done + active if sum(s == "det" for s in t.src.values()) >= min_hits]


def tracks_to_dets(tracks, high=0.6):
    out = defaultdict(list)
    for t in tracks:
        s = max(t.score, high)
        for fr, d in t.boxes.items():
            e = d.copy()
            e[4] = s
            out[fr].append(e)
    return {k: np.array(v) for k, v in out.items()}


def gaps(t, tail=0):
    """Các frame trống bên trong track (và `tail` frame sau frame cuối)."""
    fs = sorted(t.boxes)
    inner = [f for f in range(fs[0], fs[-1] + 1) if f not in t.boxes]
    return inner + [f for f in range(fs[-1] + 1, fs[-1] + 1 + tail)]


def clone(tracks):
    import copy
    return copy.deepcopy(tracks)


# =========================================================== bước 5: ROI re-detection
def roi_redetect(tracks, video, det, thr=0.3, min_iou=0.3, tail=5):
    tracks = clone(tracks)
    need = defaultdict(list)
    for t in tracks:
        for f in gaps(t, tail):
            if f in t.pred:
                need[f].append(t)
    for fr, img in iter_frames(video, set(need)):
        for t in need[fr]:
            pb = t.pred[fr]
            R = detect_in_roi(det, img, pb, thr=thr)
            if len(R):
                o = iou_matrix(pb[None], R)[0]
                j = int(np.argmax(o))
                if o[j] >= min_iou:
                    t.boxes[fr], t.src[fr] = R[j], "roi"
    return tracks


# =========================================================== bước 7: optical flow
def flow_fill(tracks, video, max_prop=15, min_pts=5, fb_thr=1.0):
    """Kéo box từ frame trước sang frame trống bằng Lucas-Kanade (chỉ trong khoảng trống bên trong track)."""
    tracks = clone(tracks)
    todo = {t.id: set(gaps(t)) for t in tracks}
    if not any(todo.values()):
        return tracks
    need = set()
    for s in todo.values():
        for f in s:
            need |= {f, f - 1}
    prev, prev_fr, run = None, -1, defaultdict(int)
    for fr, img in iter_frames(video, need):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev is not None and prev_fr == fr - 1:
            for t in tracks:
                if fr not in todo[t.id] or (fr - 1) not in t.boxes or run[t.id] >= max_prop:
                    continue
                b = t.boxes[fr - 1]
                x, y, w, h = [int(v) for v in b[:4]]
                mask = np.zeros_like(prev)
                mask[max(0, y):y + h, max(0, x):x + w] = 255
                p0 = cv2.goodFeaturesToTrack(prev, 30, 0.01, 3, mask=mask)
                if p0 is None or len(p0) < min_pts:
                    continue
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None)
                ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0 - pb, axis=2)[:, 0] < fb_thr)
                if ok.sum() < min_pts:
                    continue
                dx, dy = np.median((p1 - p0)[ok, 0], axis=0)
                nb = b.copy()
                nb[0] += dx
                nb[1] += dy
                nb[5::2] += dx
                nb[6::2] += dy
                t.boxes[fr], t.src[fr] = nb, "flow"
                run[t.id] += 1
        prev, prev_fr = g, fr
    return tracks


# =========================================================== bước 6: nội suy
def interpolate(tracks, max_gap=8):
    tracks = clone(tracks)
    for t in tracks:
        fs = sorted(t.boxes)
        for a, b in zip(fs, fs[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                da, db = t.boxes[a], t.boxes[b]
                for g in range(1, gap + 1):
                    k = g / (gap + 1)
                    t.boxes[a + g] = (1 - k) * da + k * db
                    t.src[a + g] = "interp"
    return tracks


# =========================================================== bước 9: RTS smoother
def rts_smooth(tracks):
    """Kalman xuôi trên các frame có box, rồi Rauch-Tung-Striebel ngược để làm mượt."""
    tracks = clone(tracks)
    for t in tracks:
        fs = list(range(min(t.boxes), max(t.boxes) + 1))
        kf = KF(t.boxes[fs[0]][:4])
        xs, Ps, xp, Pp = [], [], [], []
        for i, f in enumerate(fs):
            if i:
                kf.predict()
            xp.append(kf.x.copy())
            Pp.append(kf.P.copy())
            if f in t.boxes:
                kf.update(t.boxes[f][:4])
            xs.append(kf.x.copy())
            Ps.append(kf.P.copy())
        for i in range(len(fs) - 2, -1, -1):
            C = Ps[i] @ KF.F.T @ np.linalg.inv(Pp[i + 1])
            xs[i] = xs[i] + C @ (xs[i + 1] - xp[i + 1])
            Ps[i] = Ps[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
        for i, f in enumerate(fs):
            if f in t.boxes:
                d = t.boxes[f]
                nb = c2xywh(xs[i][:4])
                dx, dy = nb[0] + nb[2] / 2 - (d[0] + d[2] / 2), nb[1] + nb[3] / 2 - (d[1] + d[3] / 2)
                d[:4] = nb
                d[5::2] += dx
                d[6::2] += dy
    return tracks


# =========================================================== metric
def bucket(h):
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= h < hi:
            return name
    return SIZE_BUCKETS[-1][2]


def match_frame(dets, gts):
    tp = np.zeros(len(dets), bool)
    used, ious = set(), []
    if len(dets) == 0 or len(gts) == 0:
        return tp, used, ious
    M = iou_matrix(dets, gts)
    for i in np.argsort(-dets[:, 4]):
        cand = [(M[i, j], j) for j in range(len(gts)) if j not in used and M[i, j] >= IOU_MATCH]
        if cand:
            o, j = max(cand)
            used.add(j)
            tp[i] = True
            ious.append(o)
    return tp, used, ious


def evaluate(dets, gt, thr):
    frames = set(gt) | set(dets)
    TP = FP = FN = 0
    bt, bh = defaultdict(int), defaultdict(int)
    scored, all_iou, n_gt = [], [], 0
    E = np.zeros((0, 15))
    for fr in frames:
        G = gt.get(fr, E)
        n_gt += len(G)
        Dall = dets.get(fr, E)
        tpa, _, _ = match_frame(Dall, G)
        scored += list(zip(Dall[:, 4], tpa))
        D = Dall[Dall[:, 4] >= thr] if len(Dall) else Dall
        tp, used, ious = match_frame(D, G)
        TP += int(tp.sum())
        FP += len(D) - int(tp.sum())
        FN += len(G) - len(used)
        all_iou += ious
        for j, g in enumerate(G):
            b = bucket(g[3])
            bt[b] += 1
            bh[b] += j in used
    P, R = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    scored.sort(key=lambda x: -x[0])
    tps = np.cumsum([t for _, t in scored]) if scored else np.zeros(1)
    fps = np.cumsum([not t for _, t in scored]) if scored else np.zeros(1)
    rec, prec = tps / max(n_gt, 1), tps / np.maximum(tps + fps, 1)
    mrec, mpre = np.r_[0, rec, 1], np.r_[0, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return dict(P=P, R=R, F1=2 * P * R / max(P + R, 1e-9), AP=ap, TP=TP, FP=FP, FN=FN,
                mIoU=float(np.mean(all_iou)) if all_iou else 0.0,
                by_size={b: bh[b] / bt[b] for b in bt})


# =========================================================== pipeline
def cmd_eval(a):
    raw = load_mot(a.raw, has_score=True)
    gt = load_mot(a.gt, has_score=False)
    persons = load_mot(a.person, has_score=False) if a.person else None
    det = YuNet(a.model, parse_size(a.size)) if (a.model and a.video) else None
    nms = {"weighted": weighted_nms, "soft": soft_nms}[a.nms]
    V = {}

    V["0. baseline (NMS + thr)"] = {f: hard_nms(D, 0.3) for f, D in raw.items()}
    s1 = {f: nms(D) for f, D in raw.items()}
    V[f"1. + {a.nms} NMS"] = s1
    s2 = {f: np.array([d for d in D if geometry_ok(d)
                       and (persons is None or in_head_region(d, persons.get(f, [])))]).reshape(-1, 15)
          for f, D in s1.items()}
    V["2. + geometry" + ("/person" if persons else "")] = s2
    if det:
        s3 = verify(s2, a.video, det, low=a.low, high=a.thr)
        V["3. + verifier (đổi tỉ lệ)"] = s3
    else:
        s3 = s2
        print("(bỏ qua bước 3, 5, 6 vì thiếu --video/--model)")
    tr = track(s3, high=a.thr, low=a.low)
    V["4. + Kalman tracker"] = tracks_to_dets(tr, a.thr)
    if det:
        tr = roi_redetect(tr, a.video, det, thr=a.low)
        V["5. + ROI re-detect"] = tracks_to_dets(tr, a.thr)
        tr = flow_fill(tr, a.video)
        V["6. + optical flow"] = tracks_to_dets(tr, a.thr)
    tr = interpolate(tr, a.max_gap)
    V[f"7. + interp gap<={a.max_gap}"] = tracks_to_dets(tr, a.thr)
    tr = rts_smooth(tr)
    V["8. + RTS smoothing"] = tracks_to_dets(tr, a.thr)

    names = [b[2] for b in SIZE_BUCKETS]
    hdr = f"{'variant':28s} {'P':>6s} {'R':>6s} {'F1':>6s} {'AP':>6s} {'mIoU':>6s} {'FP':>5s} {'FN':>5s}  " \
          + " ".join(f"R{n:>8s}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    for i, (name, d) in enumerate(V.items()):
        m = evaluate(d, gt, a.thr)
        print(f"{name:28s} {m['P']:6.3f} {m['R']:6.3f} {m['F1']:6.3f} {m['AP']:6.3f} {m['mIoU']:6.3f} "
              f"{m['FP']:5d} {m['FN']:5d}  "
              + " ".join(f"{m['by_size'].get(n, float('nan')):9.3f}" for n in names))
        if a.out_dir:
            save_mot(d, os.path.join(a.out_dir, f"variant_{i}.txt"))
    src = defaultdict(int)
    for t in tr:
        for s in t.src.values():
            src[s] += 1
    print("\nNguồn box ở kết quả cuối:", dict(src))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--video", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--out", default="raw.txt")
    d.add_argument("--size", help='cạnh dài đưa vào model (vd 640, 1280) hoặc "native"')
    e = sub.add_parser("eval")
    e.add_argument("--raw", required=True)
    e.add_argument("--gt", required=True)
    e.add_argument("--video")
    e.add_argument("--model")
    e.add_argument("--size", help="giống --size của detect (dùng cho verifier/ROI)")
    e.add_argument("--person", help="person box (MOT) cho ràng buộc vùng đầu")
    e.add_argument("--nms", choices=["weighted", "soft"], default="weighted")
    e.add_argument("--thr", type=float, default=0.6, help="ngưỡng score cao (P/R + track)")
    e.add_argument("--low", type=float, default=0.3, help="ngưỡng score thấp (hysteresis, re-detect)")
    e.add_argument("--max-gap", type=int, default=8)
    e.add_argument("--out-dir", help="lưu kết quả từng biến thể (MOT) để xem lại")
    args = ap.parse_args()
    cmd_detect(args) if args.cmd == "detect" else cmd_eval(args)
