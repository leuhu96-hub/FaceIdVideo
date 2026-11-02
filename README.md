# Face ID Video (Android, Java)

Chọn một video → app phân tích các khuôn mặt → gán **ID cố định cho từng người trong suốt video** → phát lại video với khung và nhãn `ID n` trên mỗi khuôn mặt. Chạm vào một khuôn mặt (hoặc ảnh đại diện ở hàng dưới) để chọn người đó: người được chọn được tô đậm, timeline hiện các đoạn người đó xuất hiện, chạm vào timeline để tua đến đoạn đó.

## Pipeline

```
MediaCodec decode (~6 fps, ≤960 px)  →  phát hiện chuyển cảnh (histogram RGB)
  →  ML Kit Face Detection (box + 5 landmark)  →  căn chỉnh 5 điểm về 112×112 (template ArcFace)
  →  điểm chất lượng (kích thước × độ chính diện)  →  embedding TFLite (EdgeFace/MobileFaceNet…)
  →  tracker IoU + appearance (cắt track tại chuyển cảnh)
  →  gom track thành người: agglomerative average-linkage + cannot-link (2 track cùng khung ≠ 1 người)
  →  phát lại bằng ExoPlayer + overlay nội suy box theo thời gian phát
```

Phần lõi nằm trong `core/` (Java thuần, không phụ thuộc Android, có unit test):
`FaceTracker`, `TrackClusterer`, `VideoResult`, `HistogramCut`, `SimilarityTransform`.

## Chạy

1. Mở thư mục này bằng Android Studio (Ladybug trở lên) → Sync Gradle → Run lên máy thật (minSdk 26).
   Nếu Studio báo thiếu Gradle wrapper, chọn *Use Gradle wrapper* / để Studio tự tạo (Gradle 8.10.2).
2. **Thêm model embedding** vào `app/src/main/assets/face_embedder.tflite`:
   - EdgeFace (khuyến nghị): chạy `tools/convert_edgeface_to_tflite.py` (cần `torch`, `ai-edge-torch`).
     Trên Windows (`ai-edge-torch` không hỗ trợ), dùng `tools/convert_edgeface_to_tflite_windows.py` (qua ONNX + `onnx2tf`).
   - Hoặc bất kỳ model nhận dạng mặt TFLite **float32** nào, input `[1,H,W,3]` hoặc `[1,3,H,W]`.
     App tự đọc kích thước input/output. Nhớ chỉnh `Config.INPUT_MEAN/INPUT_STD` theo model
     (EdgeFace/ArcFace: 127.5/127.5; MobileFaceNet: 127.5/128).
   - Không có model, app vẫn chạy ở chế độ **chỉ tracking**: ID chỉ đúng trong từng cảnh, không nối được người qua các lần cắt cảnh.

## Sử dụng

- **Chọn video**: dùng Photo Picker, không cần quyền truy cập bộ nhớ.
- **Chạm vào khuôn mặt hoặc ảnh đại diện** để chọn hoặc bỏ chọn người đó.
- **Chỉ hiện người đã chọn**: ẩn khung của những người khác.
- **Ngưỡng gom nhóm**: kéo xong là app gom lại ngay, không cần phân tích lại video. Cao hơn nghĩa là tách người chặt hơn (ít gộp nhầm, dễ bị tách một người thành nhiều ID); thấp hơn thì ngược lại.
- **Gộp…**: chọn người A, bấm *Gộp…*, rồi chạm vào người B; B sẽ được gộp vào A. Dùng khi một người bị tách thành 2 ID. Lưu ý: kéo thanh ngưỡng sẽ xoá các lần gộp tay.

## Tinh chỉnh (`Config.java`)

| Tham số | Mặc định | Ảnh hưởng |
|---|---|---|
| `SAMPLE_INTERVAL_MS` | 166 (~6 fps) | Nhỏ hơn → track mượt/ít đứt hơn, phân tích lâu hơn |
| `ANALYSIS_MAX_SIDE` | 960 | Lớn hơn → bắt được mặt nhỏ, chậm hơn |
| `MIN_EMBED_FACE_PX` | 40 | Mặt nhỏ hơn không tính embedding (chỉ track) |
| `DEFAULT_CLUSTER_THRESHOLD` | 0.45 | Phụ thuộc model; nên chỉnh trên vài video có ground truth |
| `SCENE_CUT_THRESHOLD` | 0.35 | Thấp hơn → cắt track nhiều hơn (an toàn, clustering sẽ nối lại) |

## Giới hạn đã biết

- ML Kit chỉ là detector; mặt quá nhỏ (dưới khoảng 40–50 px trong khung phân tích) hoặc quay nghiêng mạnh sẽ bị bỏ sót hoặc không có embedding.
- Video HDR 10-bit (HEVC Main10, P010) có thể không trả về `YUV_420_888` trên một số decoder.
- Kết quả chỉ nằm trong bộ nhớ, chưa lưu lại. Cần phân tích lại khi mở lại video.
- Embedding chạy trên CPU (XNNPACK). Muốn dùng NPU Exynos thì chuyển sang LiteRT CompiledModel.
- Dữ liệu khuôn mặt là dữ liệu sinh trắc học: mọi thứ đều xử lý trên máy và không gửi đi đâu. Khi phát hành, cần xin đồng ý của người dùng (Luật BVDLCN 91/2025/QH15).

## Đánh giá

Để đo độ chính xác, gán nhãn track bằng CVAT, rồi tính BCubed F và WCP ở mức track, cùng sai số số người |K̂−K|, so với kết quả của `TrackClusterer`.
