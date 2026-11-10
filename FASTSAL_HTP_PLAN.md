# PLAN: Chuyển FastSal → DLC chạy trên HTP (Snapdragon sm8950) với QAIRT SDK 2.51

> Tài liệu này dành cho coding agent (Gemini CLI). Thực hiện tuần tự từng Phase.
> Mỗi Phase có **Tiêu chí hoàn thành**; chưa đạt thì KHÔNG sang Phase tiếp theo.

---

## 0. CONFIG — người dùng điền trước khi chạy

```yaml
FASTSAL_REPO:     <đường dẫn repo FastSal, vd D:/work/FastSal>
CHECKPOINT:       <file .pth, vd D:/work/FastSal/weights/fastsal_c.pth>
MODEL_VARIANT:    <A | C | ... — biến thể FastSal đang dùng>
INPUT_H, INPUT_W: <vd 192, 256 — đúng size lúc train>
QAIRT_ROOT_WSL:   <vd /home/<user>/qairt/2.51.x>   # SDK cài trong WSL2
WORK_DIR:         <vd D:/work/fastsal_htp>        # thư mục làm việc mới
TEST_IMAGES_DIR:  <thư mục ~5 ảnh test>
CALIB_IMAGES_DIR: <thư mục 100–300 ảnh calibration (chỉ cần nếu làm A16W8)>
TARGET_SOC:       sm8950
DEVICE_DIR:       /data/local/tmp/fastsal
```

Nếu giá trị nào còn `<...>`: tự tìm trong repo/máy; không tìm được thì DỪNG và hỏi người dùng.

---

## 1. Quy tắc cho agent

1. **Môi trường**: máy host là Windows.
   - Python/PyTorch/ONNX export: chạy trên Windows hoặc WSL đều được.
   - Toàn bộ lệnh `qairt-*`, `snpe-*` (host tools): chạy trong **WSL2** sau khi `source $QAIRT_ROOT_WSL/bin/envsetup.sh`.
   - `adb`: chạy từ Windows (PowerShell). Nếu dùng Git Bash, PHẢI `export MSYS_NO_PATHCONV=1` trước mọi lệnh adb có đường dẫn `/data/...`.
2. **Không sửa** code model gốc trong `FASTSAL_REPO`. Mọi script mới đặt trong `WORK_DIR/scripts/`.
3. **Không cài/đổi phiên bản SDK.** Tất cả tool và `.so` lấy từ cùng QAIRT 2.51.
4. Mọi lệnh đã chạy + output quan trọng ghi vào `WORK_DIR/logs/<phase>.log`.
5. Không đoán flag CLI: trước khi dùng một tool, chạy `<tool> --help` và đối chiếu. Nếu flag trong plan không tồn tại ở 2.51, dùng flag tương đương và ghi chú lại trong `REPORT.md`.
6. Gặp lỗi: thử tối đa 2 cách sửa hợp lý, sau đó DỪNG, ghi lỗi + những gì đã thử vào `REPORT.md` và hỏi người dùng.

---

## 2. Cấu trúc thư mục

```
WORK_DIR/
  scripts/      export_onnx.py, make_raw.py, convert.sh, device_run.ps1, compare.py
  models/       fastsal.onnx, fastsal_sim.onnx, *.dlc
  data/
    test_raw/   *.raw + input_list.txt (test)
    calib_raw/  *.raw + calib_list.txt (calibration)
    ref/        output tham chiếu từ PyTorch/ONNX (.npy)
  device_out/   output kéo về từ điện thoại
  logs/
  REPORT.md
```

---

## Phase 1 — Khảo sát model

**Việc cần làm**
- Đọc `FASTSAL_REPO`: tìm class model, cách load checkpoint, biến thể `MODEL_VARIANT`.
- Xác định: input size, preprocessing (resize, chia 255, mean/std, thứ tự kênh RGB/BGR), output (shape, có sigmoid/normalize ở cuối không), postprocessing.
- Chỉ dùng mạng student/inference; bỏ teacher, loss, phần train.
- Ghi tất cả vào `REPORT.md` mục "Model info".

**Tiêu chí hoàn thành**
- Load được checkpoint, chạy forward 1 ảnh test trên PyTorch (CPU) không lỗi.
- `REPORT.md` có đủ: input shape, preprocessing, output shape, postprocessing.

---

## Phase 2 — Export ONNX FP32 & kiểm chứng

**Script `scripts/export_onnx.py`**
- `model.eval()`, input giả `1×3×INPUT_H×INPUT_W`, `torch.onnx.export(..., opset_version=17, do_constant_folding=True, input_names=["input"], output_names=["saliency"])`, **shape tĩnh** (không dynamic_axes).
- Sau export: `onnxsim fastsal.onnx fastsal_sim.onnx` (nếu chưa có `onnxsim` thì báo người dùng, không tự cài nếu không có mạng).
- Chạy cùng 5 ảnh test qua PyTorch và onnxruntime; lưu output PyTorch vào `data/ref/<name>.npy`.

**Tiêu chí hoàn thành**
- `max_abs_diff(PyTorch, ORT) < 1e-4` và `CC > 0.9999` trên cả 5 ảnh.
- Ghi danh sách op của ONNX (đếm theo loại) vào `REPORT.md`.

---

## Phase 3 — Convert sang DLC

**Script `scripts/convert.sh` (chạy trong WSL)**

3a. FP16 (bắt buộc):
```bash
qairt-converter --input_network models/fastsal_sim.onnx \
  --source_model_input_shape input 1,3,${INPUT_H},${INPUT_W} \
  --float_bitwidth 16 \
  --output_path models/fastsal_fp16.dlc
```

3b. A16W8 (chỉ làm nếu `CALIB_IMAGES_DIR` có ảnh):
```bash
qairt-converter --input_network models/fastsal_sim.onnx \
  --source_model_input_shape input 1,3,${INPUT_H},${INPUT_W} \
  --output_path models/fastsal_fp32.dlc
qairt-quantizer --input_dlc models/fastsal_fp32.dlc \
  --input_list data/calib_raw/calib_list.txt \
  --act_bitwidth 16 --weights_bitwidth 8 \
  --output_dlc models/fastsal_a16w8.dlc
```

3c. Sau mỗi DLC: `qairt-dlc-info -i <dlc>` → lưu vào `logs/`. Ghi vào `REPORT.md`: tên + shape + **layout** (NHWC/NCHW) của input/output trong DLC, dtype các tensor, cảnh báo op không hỗ trợ HTP.

**Script `scripts/make_raw.py`** (cần trước 3b và Phase 5)
- Tham số: thư mục ảnh, thư mục output, layout (`nhwc`|`nchw`).
- Áp đúng preprocessing ở Phase 1, ghi float32 `.tofile()` theo **layout mà DLC yêu cầu** (đọc từ `qairt-dlc-info`).
- Sinh `input_list.txt` / `calib_list.txt` (mỗi dòng một file; đường dẫn tương đối với thư mục sẽ chạy).
- Đồng thời lưu bản NCHW `.npy` để chạy ORT.
- Kiểm tra: số byte mỗi file = H×W×3×4.

**Tiêu chí hoàn thành**
- Có `fastsal_fp16.dlc` (và `fastsal_a16w8.dlc` nếu làm 3b), convert không có ERROR.

---

## Phase 4 — Offline graph cache cho sm8950

```bash
snpe-dlc-graph-prepare --input_dlc models/fastsal_fp16.dlc \
  --output_dlc models/fastsal_fp16_sm8950.dlc --htp_socs sm8950
```
(lặp lại cho a16w8 nếu có)

- Trước tiên kiểm tra `snpe-dlc-graph-prepare --help` và tài liệu SDK (`docs/`, bảng Supported Snapdragon Devices) xem 2.51 có hỗ trợ `sm8950` không, và sm8950 ứng với Hexagon **V??** nào. Ghi `HEXAGON_ARCH` vào `REPORT.md`.
- Nếu `sm8950` không được hỗ trợ: KHÔNG thử SoC khác. Ghi lại, bỏ qua Phase 4, dùng DLC không cache ở Phase 5 (sẽ online prepare trên máy).

**Tiêu chí hoàn thành**
- Có DLC đã cache, HOẶC đã ghi rõ lý do bỏ qua + `HEXAGON_ARCH` xác định được.

---

## Phase 5 — Chạy trên thiết bị qua adb

**Kiểm tra trước**
- `adb devices` thấy máy; `adb shell getprop ro.soc.model` → ghi vào REPORT (phải khớp `TARGET_SOC`, nếu không thì DỪNG và hỏi).
- `adb shell ls -l /vendor/lib64/libcdsprpc.so` tồn tại.

**Script `scripts/device_run.ps1`** (PowerShell) thực hiện:
1. `adb shell mkdir -p DEVICE_DIR`
2. Push từ SDK:
   - `bin/aarch64-android/snpe-net-run`, `bin/aarch64-android/snpe-platform-validator`
   - toàn bộ `lib/aarch64-android/`
   - toàn bộ `lib/hexagon-<HEXAGON_ARCH>/unsigned/`
   - các DLC, `data/test_raw/*`
3. `chmod +x` các binary.
4. Chạy trong một lệnh `adb shell "cd DEVICE_DIR && export LD_LIBRARY_PATH=$PWD && export ADSP_LIBRARY_PATH='$PWD;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && ..."`:
   - `./snpe-platform-validator --runtime dsp`
   - Với mỗi DLC: `./snpe-net-run --container <dlc> --input_list input_list.txt --use_cpu --output_dir out_<tên>_cpu`
   - Với mỗi DLC: `./snpe-net-run --container <dlc> --input_list input_list.txt --use_dsp --perf_profile burst --output_dir out_<tên>_dsp`
5. `adb pull` tất cả thư mục `out_*` về `device_out/`.
6. Lưu stdout/stderr vào `logs/phase5.log`.

**Nếu `--use_dsp` lỗi** — chẩn đoán theo thứ tự, ghi kết quả từng bước:
1. CPU có chạy không? (không → lỗi ở convert, quay lại Phase 3)
2. `snpe-platform-validator --runtime dsp` có PASS không? (không → sai Stub/Skel V?? hoặc ADSP_LIBRARY_PATH)
3. DLC không cache có chạy DSP không? (có → cache không khớp)
4. `adb logcat -d | grep -iE "snpe|qnn|cdsprpc|fastrpc|htp"` → trích lỗi vào REPORT.

**Tiêu chí hoàn thành**
- Ít nhất một DLC chạy thành công với `--use_dsp`, có file output trong `device_out/`.

---

## Phase 6 — So sánh độ chính xác & hiệu năng

**Script `scripts/compare.py`**
- Đọc `device_out/out_*/Result_i/<output>.raw` (float32), reshape theo shape output (đọc từ dlc-info; chuyển NHWC→NCHW nếu cần).
- So với `data/ref/*.npy` (PyTorch): max_abs_diff, mean_abs_diff, Pearson CC, KLD (sau khi chuẩn hoá thành phân phối), có NaN/Inf không.
- Xuất ảnh so sánh `ref | cpu | dsp` (PNG) cho từng ảnh test.
- In bảng tổng hợp theo từng DLC × runtime.

**Hiệu năng**
- Trong WSL: `snpe-diagview --input_log device_out/out_<tên>_dsp/SNPEDiag.log` → lấy thời gian init và inference trung bình.

**Tiêu chí hoàn thành**
- FP16 trên DSP: CC ≥ 0.99, không NaN. A16W8: CC ≥ 0.98 (nếu thấp hơn → ghi chú, đề xuất nguyên nhân, KHÔNG tự đổi chiến lược quantize).
- Bảng kết quả + thời gian có trong `REPORT.md`.

**Nếu sai số lớn / NaN**
- Chạy lại DSP với `--debug` để dump output từng layer; so với output trung gian từ onnxruntime (thêm các tensor trung gian làm output tạm trong một bản ONNX copy) để tìm layer đầu tiên lệch. Ghi kết quả vào REPORT, rồi DỪNG và hỏi người dùng.

---

## 7. Deliverables cuối cùng

- [ ] `models/fastsal_sim.onnx`
- [ ] `models/fastsal_fp16*.dlc` (+ `fastsal_a16w8*.dlc` nếu có)
- [ ] Toàn bộ `scripts/` chạy lại được từ đầu
- [ ] `REPORT.md` gồm: Model info, lệnh đã dùng, layout IO của DLC, HEXAGON_ARCH, danh sách `.so` cần đóng gói vào app (`jniLibs/arm64-v8a`), bảng accuracy, bảng thời gian, vấn đề tồn đọng.

## 8. Ngoài phạm vi (không làm)

- Tích hợp vào app Android.
- Train lại / fine-tune FastSal.
- Thay đổi kiến trúc model.
