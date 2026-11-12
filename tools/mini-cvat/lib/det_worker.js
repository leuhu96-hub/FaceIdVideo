/* Web Worker cho "Tự gán nhãn": mỗi worker giữ một bản OpenCV.js + các model, xử lý song song theo frame.
   Tin nhắn vào:
     {t:'init', models:{det_500m:Uint8Array,...}}                      -> {t:'ready'}
     {t:'det', id, w, h, buf(RGBA), cfg}                                -> {id, dets:[{x,y,w,h,s,lm,n,g}]}
     {t:'emb', id, w, h, buf(RGBA), lms:[[10 số],...]}                  -> {id, embs:[Float32Array(512),...]}
   Lỗi trả về {id, error}. */
let core = null, ready = null, models = {};
function boot() {
  if (ready) return ready;
  ready = new Promise((res, rej) => {
    try { importScripts('opencv.js', 'yunet_model.js', 'det_core.js'); } catch (e) { rej(e); return; }
    const t0 = Date.now();
    const poll = () => {
      if (self.cv && typeof self.cv.Mat === 'function') {
        try {
          const bin = atob(self.YUNET_B64), u8 = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
          core = DetCore.make(self.cv, { yunetBytes: u8, models }); res(true);
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
    if (m.t === 'init') { Object.assign(models, m.models || {}); await boot(); for (const k in m.models || {}) core.addModel(k, m.models[k]); self.postMessage({ t: 'ready' }); return; }
    await boot();
    if (m.t === 'det') { const dets = core.detect(m.w, m.h, m.buf, m.cfg); self.postMessage({ id: m.id, dets }, dets.map(d => d.g.buffer)); }
    else if (m.t === 'emb') { const embs = core.embed(m.w, m.h, m.buf, m.lms); self.postMessage({ id: m.id, embs }, embs.map(x => x.buffer)); }
  } catch (err) { self.postMessage({ id: m.id, error: String((err && err.message) || err) }); }
};
