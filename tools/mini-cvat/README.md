# Mini CVAT Video

Công cụ gán nhãn video chạy hoàn toàn trong trình duyệt, kèm đánh giá ground truth (GT) và đánh giá model face detection (YuNet). Video và nhãn không rời khỏi máy.

## Chạy

```
git clone https://github.com/leuhu96-hub/FaceIdVideo
cd FaceIdVideo/tools/mini-cvat
python -m http.server 8000
```

Mở http://localhost:8000 bằng Chrome hoặc Edge, bấm **Mở video** (MP4 H.264 hoặc WebM).

Mở thẳng `index.html` (double-click) vẫn chạy được, nhưng Web Worker bị chặn nên video có thể giật khi chạy YuNet. Nên dùng web server như trên.

## Cấu trúc

| Đường dẫn | Nội dung |
| --- | --- |
| `index.html` | Công cụ gán nhãn, Đánh giá GT, Box model, Đánh giá model so với GT |
| `lib/opencv.js` | OpenCV.js 4.10 (Apache 2.0), chỉ tải khi chạy nhóm C hoặc Box model |
| `lib/yunet_model.js` | YuNet 2026may (MIT), nhúng dạng base64 |
| `lib/md_worker.js` | Web Worker chạy YuNet |
| `sample/` | Video thử `c1.webm` (2 người, có cài sẵn lỗi GT) và nhãn `c1_gt.json` |
| `scripts/analyze_gt.py` | Phân tích GT đầy đủ, xuất báo cáo HTML + CSV + tham số gợi ý |
| `scripts/eval_face_postproc.py` | Chạy YuNet, so sánh 9 biến thể post-processing với GT |
| `scripts/make_test_video.py` | Tạo video thử có lỗi GT cài sẵn |

## Định dạng hỗ trợ

| Định dạng | Nhập GT (nút **Nhập**) | Nhập kết quả model (**Nhập kết quả…**) | Xuất |
| --- | --- | --- | --- |
| CVAT for video 1.1 (.xml / .zip) | có | – | có |
| Mini CVAT JSON | có | – | có |
| Music Video XML (`<Video><Trajectory obj_id><Frame …/>`) | có | có | có |
| JSONL (mỗi dòng một frame) | có | có | – |
| JSON theo frame (`data.frames`) | có | có | – |
| JSON mảng bản ghi kiểu JSONL (`[{"sample_id":…,"object":[…]}]` hoặc `{"samples":[…]}`) | có | có | – |
| MOT 1.1 (`frame,id,x,y,w,h,score[,10 landmark]`) | – | có | có |
| YOLO | – | – | có |

Số frame trong công cụ đếm từ 0; file MOT đếm từ 1.

### JSONL

```
{"sample_id":"video/frame-000011","object":[{"track_id":"1","bbox":[1,2,3,4],"score":0.93}]}
```

- Số frame lấy từ số cuối trong `sample_id`; phần trước `/` là tên video. `score` không bắt buộc.
- Khi nhập, công cụ hỏi định dạng bbox (`[x,y,w,h]`, `[x1,y1,x2,y2]`, `[cx,cy,w,h]`, pixel hay chuẩn hoá 0–1), frame đầu là `frame-000000` hay `frame-000001`, và video nào nếu file có nhiều video. Các lựa chọn được đoán sẵn; nếu đã mở GT, công cụ thử mọi tổ hợp và chọn tổ hợp khớp GT nhiều nhất.
- Dòng JSON hơi lỗi (thiếu dấu `"`) vẫn được đọc theo mẫu. `object` rỗng = frame không có mặt.

### JSON theo frame (kết quả detector)

```
{"data":{"detector":{"inputResolution":"original","model":"yunet"},
         "frames":{"1":[{"bbox":[1,2,3,4],"confidence":0.5,"id":"f1_1","landmarks":[1,2,3,4,5,6,7,8,9,10]}]}}}
```

- Khoá của `frames` là số frame; `confidence` là score; `landmarks` (5 điểm x,y) được vẽ lên box. Tên model và `inputResolution` hiện ở mục Box model.
- Cũng nhận `frames` ở ngoài cùng, hoặc `frames` là mảng `[{"frame":1,"faces":[…]}]`.
- Định dạng bbox và frame đầu được hỏi và đoán sẵn giống JSONL. Lỗi gõ hay gặp (thiếu `"`, `::`, dấu phẩy thừa) được tự sửa.

## Thử nhanh

1. Mở `sample/c1.webm`, đặt FPS = 25.
2. **Nhập** → **Chọn file…** → `sample/c1_gt.json`.
3. **Đánh giá GT** → Nhãn `face` → bật Nhóm C → **Chạy phân tích**.
4. Kết quả đúng: mặt thiếu nhãn khoảng frame 20–32, box vẽ vào nền khoảng frame 80–88, (có SFace) 2 cặp "cùng người nhưng 2 ID".

## Đánh giá model so với GT

1. Mở video, **Nhập** GT.
2. Ở mục **Box model** (thanh bên phải), bấm **Nhập kết quả…** và chọn file của model.
3. Bấm **Đánh giá model so với GT…** → **Tính**.

Kết quả: P, R, F1, AP, mIoU, TP/FP/FN, % frame đúng hoàn toàn, recall theo cỡ mặt, bảng theo ngưỡng score, recall từng ID, danh sách đoạn lỗi (bấm để nhảy tới; phím `,` / `.`). **Tải CSV** ghi từng frame lỗi. Cách ghép box giống dòng `0. baseline` của `eval_face_postproc.py eval`.

## Đánh giá cả folder

Nút **Đánh giá folder** trên thanh công cụ chấm nhiều video một lần.

1. Chọn **Folder video…**, **Folder GT…**, và **Folder kết quả model…** (không bắt buộc).
2. Công cụ ghép theo tên file, không phân biệt hoa thường: `videos/a.mp4` ↔ `gt/a.txt` hoặc `gt/a.json` ↔ `results/a.json`. Đuôi `_gt`, `_pred`, `_result`… được bỏ qua; file tên chung như `gt/a/gt.txt` lấy tên thư mục cha.
3. Mỗi video chỉ đọc một file GT. Nếu có cả `.txt` và `.json`, tuỳ chọn "Có cả .txt và .json thì đọc" quyết định đọc file nào.
4. GT `.txt`: MOT (`frame,id,x,y,w,h,…`, frame từ 1) hoặc JSONL. GT `.json`: JSON theo frame, mảng bản ghi, Mini CVAT JSON.
5. Bấm **Chạy**. Mỗi video chạy Đánh giá GT (nhóm A, thêm B/C nếu bật) rồi đánh giá model so với GT. Video không có file kết quả có thể chạy YuNet trong trình duyệt mỗi N frame.

Kết quả có hai phần:

- **Đánh giá model so với GT**: bảng từng video (P, R, F1, AP, FP, FN, % frame đúng hoàn toàn, định dạng đã đoán) và dòng tổng gộp TP/FP/FN của cả folder.
- **Đánh giá ground truth cả folder**: cùng các mục như báo cáo từng file nhưng gộp mọi video (kết luận nhanh, tham số gợi ý đủ cho mọi video, tổng quan từng video, cỡ mặt theo input, hình dạng, thời gian và track, chuyển động, tương tác, chất lượng ảnh, đối chiếu YuNet), bảng nghi vấn chung lọc theo loại / video, và báo cáo đầy đủ của từng video (mở rộng được). Bấm tên video để mở nó, kèm danh sách nghi vấn GT (`[` `]`) và đoạn model sai (`,` `.`). **Tải kết quả** xuất `folder_summary.csv`, `folder_model_errors.csv`, `folder_gt_report.html` (báo cáo GT cả folder + từng video), `folder_gt_issues.csv`, `folder_gt_events.csv`, `folder_gt_ids.csv`, `folder_gt_attrs.csv` (khi bật nhóm B) và `folder_suggested_params.json` (tham số chung + từng video).

Video mở từ Đánh giá folder **không tự lưu nhãn**, để không ghi đè nhãn bạn đã lưu cho video cùng tên. Sửa GT xong hãy **Xuất**.

## Box model

- **Chạy frame này**: YuNet trên frame hiện tại. **Tự chạy YuNet (tính trước)**: một bản video ẩn detect trước đầu phát nên box không trễ.
- Màu: xanh = khớp GT; hồng "thừa" = model thấy, GT không có; vàng "model sót" = GT có, model không thấy; box mờ "ước lượng từ fN" = frame chưa chạy, giữ box frame gần nhất và dời theo chuyển động.
- **Thêm box chưa có nhãn → GT**: biến box hồng ở frame này thành track mới.

## Script Python

```
pip install numpy opencv-python onnxruntime scipy matplotlib
```

`analyze_gt.py` import từ `eval_face_postproc.py`, giữ hai file cùng thư mục.

```
# phân tích GT (MOT hoặc Music Video XML)
python scripts/analyze_gt.py gt.txt --videos video.mp4 --model face_detection_yunet_2026may.onnx --out gt_report

# đo YuNet + 9 biến thể post-processing
python scripts/eval_face_postproc.py detect --video video.mp4 --model face_detection_yunet_2026may.onnx --size native --out raw.txt
python scripts/eval_face_postproc.py eval --raw raw.txt --gt gt.txt --video video.mp4 --model face_detection_yunet_2026may.onnx --out-dir res
```

Model .onnx không có trong repo: YuNet và SFace tải ở [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models).

## Phím tắt chính

R vẽ box · F / D frame sau / trước (Shift: ±10) · O outside · K keyframe · B ẩn/hiện box model · `[` / `]` nghi vấn GT · `,` / `.` đoạn model sai · Ctrl+Z hoàn tác · `?` xem tất cả.

## Lưu ý

- Nhãn tự lưu trong trình duyệt theo tên file video; bấm **Xuất** định kỳ để sao lưu.
- Ảnh/video khuôn mặt là dữ liệu sinh trắc học (dữ liệu nhạy cảm theo Luật Bảo vệ dữ liệu cá nhân 91/2025). Không commit video hoặc báo cáo có mặt người thật vào repo (repo này đang public).
- Giấy phép thành phần: YuNet MIT, SFace Apache 2.0, OpenCV.js Apache 2.0. Weights InsightFace (SCRFD, ArcFace) là phi thương mại.
