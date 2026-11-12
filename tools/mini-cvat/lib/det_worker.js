/* Web Worker cho "Tự gán nhãn": mỗi worker giữ một bản OpenCV.js + các model, xử lý song song theo frame.
   Tin nhắn vào:
     {t:'init', models:{det_500m:Uint8Array,...}, ort:{main,ep,threads}|null} -> {t:'ready', bk}
     {t:'det', id, w, h, buf(RGBA), cfg}                                      -> {id, dets:[{x,y,w,h,s,lm,n,g}], bk}
     {t:'emb', id, w, h, buf(RGBA), lms:[[10 số],...]}                        -> {id, embs:[Float32Array(512),...], bk}
   bk = bộ chạy model đang dùng (onnxruntime-web hoặc OpenCV.js kèm lý do). Lỗi trả về {id, error}. */
let core = null, ready = null;
function boot(models, ortCfg) {
  if (ready) return ready;
  ready = new Promise((res, rej) => {
    try { importScripts('opencv.js', 'yunet_model.js', 'det_core.js'); } catch (e) { rej(e); return; }
    let ort = null, ortWhy = null;
    if (ortCfg) {
      try {
        importScripts('ort/' + ortCfg.main); ort = self.ort;
        if (!ort || !ort.InferenceSession) throw new Error('gói onnxruntime-web không hợp lệ');
        ort.env.wasm.wasmPaths = new URL('ort/', self.location.href).href;
        ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, ortCfg.threads | 0) : 1;
        if (ortCfg.ep === 'webgpu' && !(self.navigator && self.navigator.gpu)) ortCfg.ep = 'wasm';
      } catch (e) { ort = null; ortWhy = 'không nạp được: ' + String((e && e.message) || e).slice(0, 120); }
    }
    const t0 = Date.now();
    const poll = () => {
      if (self.cv && typeof self.cv.Mat === 'function') {
        try {
          const bin = atob(self.YUNET_B64), u8 = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
          core = DetCore.make(self.cv, { yunetBytes: u8, models, ort: ort ? { lib: ort, ep: ortCfg.ep } : null, ortWhy });
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
  const m = e.data;
  try {
    if (m.t === 'init') { await boot(m.models || {}, m.ort || null); self.postMessage({ t: 'ready', bk: core.backend() }); return; }
    if (!core) throw new Error('Worker chưa khởi động');
    if (m.t === 'det') { const dets = await core.detect(m.w, m.h, m.buf, m.cfg); self.postMessage({ id: m.id, dets, bk: core.backend() }, dets.map(d => d.g.buffer)); }
    else if (m.t === 'emb') { const embs = await core.embed(m.w, m.h, m.buf, m.lms); self.postMessage({ id: m.id, embs, bk: core.backend() }, embs.map(x => x.buffer)); }
  } catch (err) { self.postMessage({ id: m.id, error: String((err && err.message) || err) }); }
};
