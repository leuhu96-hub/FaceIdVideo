/* Web Worker chạy YuNet (OpenCV.js) ngoài luồng chính để video không bị giật khi "Tự chạy YuNet".
   Nhận: {id, w, h, buf (RGBA ArrayBuffer), k (tỉ lệ đã thu nhỏ)}  ->  Trả: {id, rows: [[x,y,w,h, 10 landmark, score], ...]} theo toạ độ gốc */
let det = null, ready = null;
function init() {
  if (ready) return ready;
  ready = new Promise((res, rej) => {
    try { importScripts('opencv.js', 'yunet_model.js'); } catch (e) { rej(e); return; }
    const t0 = Date.now();
    // lưu ý: đối tượng cv có then() -> không resolve Promise bằng cv
    const poll = () => {
      if (self.cv && typeof self.cv.Mat === 'function') {
        try {
          const bin = atob(self.YUNET_B64), u8 = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
          try { cv.FS_unlink('/yunet.onnx'); } catch (e) {}
          cv.FS_createDataFile('/', 'yunet.onnx', u8, true, false, false);
          det = new cv.FaceDetectorYN('yunet.onnx', '', new cv.Size(320, 320), 0.3, 0.3, 5000);
          res(true);
        } catch (e) { rej(e); }
        return;
      }
      if (Date.now() - t0 > 90000) return rej(new Error('OpenCV.js khởi động quá lâu'));
      setTimeout(poll, 100);
    };
    poll();
  });
  return ready;
}
self.onmessage = async (e) => {
  const { id, w, h, buf, k } = e.data;
  try {
    await init();
    const rgba = cv.matFromImageData(new ImageData(new Uint8ClampedArray(buf), w, h)), bgr = new cv.Mat();
    cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR); rgba.delete();
    det.setInputSize(new cv.Size(w, h));
    const out = new cv.Mat(); det.detect(bgr, out);
    const rows = [];
    for (let i = 0; i < out.rows; i++) {
      const r = Array.from(out.data32F.slice(i * 15, i * 15 + 15));
      for (let j = 0; j < 14; j++) r[j] /= k;
      rows.push(r);
    }
    out.delete(); bgr.delete();
    self.postMessage({ id, rows });
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
