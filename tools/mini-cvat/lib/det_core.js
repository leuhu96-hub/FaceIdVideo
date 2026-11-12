/* Lõi phát hiện mặt cho "Tự gán nhãn": SCRFD (det_500m / det_10g), YuNet, và ArcFace MobileFaceNet (w600k_mbf) để nhúng mặt.
   Dùng chung cho Web Worker (lib/det_worker.js) và luồng chính (khi trình duyệt không cho tạo Worker).
   Chạy trên OpenCV.js 4.10 (cv.dnn). Các model SCRFD xuất ra với shape đầu vào động ('?'), OpenCV.js không suy được shape
   nên ở đây tự sửa byte ONNX: cố định đầu vào 1x3xHxW rồi mới nạp (DC.staticOnnx). Không cần gói onnx hay công cụ ngoài. */
(function (G) {
  'use strict';
  const DC = {};

  /* ---------- sửa protobuf ONNX: cố định shape đầu vào ---------- */
  function rv(b, i) { let r = 0, m = 1; for (;;) { const c = b[i++]; r += (c & 0x7f) * m; m *= 128; if (c < 0x80) return [r, i]; } }
  function wv(n) { const o = []; for (;;) { const c = n % 128; n = Math.floor(n / 128); if (n) o.push(c | 128); else { o.push(c); return o; } } }
  function fields(b) {
    const out = []; let i = 0;
    while (i < b.length) {
      let k; [k, i] = rv(b, i); const f = Math.floor(k / 8), w = k & 7; let v;
      if (w === 0) [v, i] = rv(b, i);
      else if (w === 2) { let l; [l, i] = rv(b, i); v = b.subarray(i, i + l); i += l; }
      else if (w === 1) { v = b.subarray(i, i + 8); i += 8; }
      else if (w === 5) { v = b.subarray(i, i + 4); i += 4; }
      else throw new Error('ONNX: wire type ' + w);
      out.push([f, w, v]);
    }
    return out;
  }
  function enc(fl) {
    const parts = []; let n = 0;
    const push = a => { const u = a instanceof Uint8Array ? a : Uint8Array.from(a); parts.push(u); n += u.length; };
    for (const [f, w, v] of fl) {
      push(wv(f * 8 + w));
      if (w === 0) push(wv(v)); else if (w === 2) { push(wv(v.length)); push(v); } else push(v);
    }
    const o = new Uint8Array(n); let p = 0; for (const u of parts) { o.set(u, p); p += u.length; } return o;
  }
  function setInputShape(vi, dims) {
    return enc(fields(vi).map(([f, w, v]) => {
      if (f !== 2) return [f, w, v];                                // ValueInfoProto.type
      return [2, 2, enc(fields(v).map(([f2, w2, v2]) => {
        if (f2 !== 1) return [f2, w2, v2];                          // TypeProto.tensor_type
        return [1, 2, enc(fields(v2).map(([f3, w3, v3]) => {
          if (f3 !== 2) return [f3, w3, v3];                        // Tensor.shape
          const dm = dims.map(x => [1, 2, Uint8Array.from([8, ...wv(x)])]);
          return [2, 2, enc(dm)];
        }))];
      }))];
    }));
  }
  DC.staticOnnx = function (bytes, dims) {
    return enc(fields(bytes).map(([f, w, v]) => {
      if (f !== 7) return [f, w, v];                                // ModelProto.graph
      const g = [];
      for (const [f2, w2, v2] of fields(v)) {
        if (f2 === 11) g.push([11, 2, setInputShape(v2, dims)]);    // graph.input
        else if (f2 === 13) continue;                               // bỏ value_info (shape suy ra cũ, có chiều '?')
        else g.push([f2, w2, v2]);
      }
      return [7, 2, enc(g)];
    }));
  };

  /* ---------- hình học ---------- */
  const iouXY = (a, b) => {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
    const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1); return i / (a.w * a.h + b.w * b.h - i + 1e-9);
  };
  DC.nms = function (L, thr) { const out = []; [...L].sort((a, b) => b.s - a.s).forEach(d => { if (out.every(o => iouXY(o, d) < thr)) out.push(d); }); return out; };
  // gộp box của nhiều bộ dò: cụm theo IoU, toạ độ lấy trung bình có trọng số điểm, điểm lấy max, ghi số bộ dò đồng ý (n)
  DC.fuse = function (L, thr) {
    const out = []; [...L].sort((a, b) => b.s - a.s).forEach(d => {
      const c = out.find(o => iouXY(o.top, d) >= thr);
      if (c) c.m.push(d); else out.push({ top: d, m: [d] });
    });
    return out.map(({ top, m }) => {
      let sw = 0, x = 0, y = 0, w = 0, h = 0; const lm = new Array(10).fill(0);
      for (const d of m) { sw += d.s; x += d.x * d.s; y += d.y * d.s; w += d.w * d.s; h += d.h * d.s; for (let j = 0; j < 10; j++) lm[j] += top.lm[j] === undefined ? 0 : d.lm[j] * d.s; }
      return { x: x / sw, y: y / sw, w: w / sw, h: h / sw, s: top.s, lm: lm.map(v => v / sw), n: m.length };
    });
  };

  /* ---------- căn chỉnh mặt về 112x112 (mẫu ArcFace) ---------- */
  const ARC = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
  DC.simMatrix = function (lm) { // 5 điểm (mắt trái, mắt phải, mũi, miệng trái, miệng phải theo toạ độ ảnh) -> ma trận 2x3
    const sx = [], sy = []; for (let j = 0; j < 5; j++) { sx.push(lm[2 * j]); sy.push(lm[2 * j + 1]); }
    const mx = sx.reduce((a, b) => a + b) / 5, my = sy.reduce((a, b) => a + b) / 5, dx = ARC.reduce((a, p) => a + p[0], 0) / 5, dy = ARC.reduce((a, p) => a + p[1], 0) / 5;
    let num1 = 0, num2 = 0, den = 0;
    for (let j = 0; j < 5; j++) { const ax = sx[j] - mx, ay = sy[j] - my, bx = ARC[j][0] - dx, by = ARC[j][1] - dy; num1 += ax * bx + ay * by; num2 += ax * by - ay * bx; den += ax * ax + ay * ay; }
    const a = num1 / (den + 1e-9), b = num2 / (den + 1e-9);
    return [a, -b, dx - (a * mx - b * my), b, a, dy - (b * mx + a * my)];
  };

  DC.make = function (cv, opts) {
    const st = { nets: new Map(), yu: null, yuLo: -1, models: opts.models || {}, arc: null };
    const write = (name, bytes) => { try { cv.FS_unlink('/' + name); } catch (e) { } cv.FS_createDataFile('/', name, bytes, true, false, false); };
    const ex = e => (typeof e === 'number' && cv.exceptionFromPtr) ? new Error(cv.exceptionFromPtr(e).msg) : e;

    function netFor(stem, W, H, dims) {
      const key = stem + '@' + W + 'x' + H; if (st.nets.has(key)) return st.nets.get(key);
      const raw = st.models[stem]; if (!raw) throw new Error('Thiếu model ' + stem + '.onnx');
      const f = key.replace('@', '_') + '.onnx';
      try {
        write(f, DC.staticOnnx(raw, dims || [1, 3, H, W]));
        const n = cv.readNet(f); n.setPreferableBackend(cv.DNN_BACKEND_OPENCV);
        st.nets.set(key, n); try { cv.FS_unlink('/' + f); } catch (e) { }
        return n;
      } catch (e) { throw ex(e); }
    }
    function yunet(lo) {
      if (!st.yu || st.yuLo !== lo) {
        if (st.yu) st.yu.delete();
        if (!st.yunetWritten) { write('yunet_core.onnx', opts.yunetBytes); st.yunetWritten = true; }
        st.yu = new cv.FaceDetectorYN('yunet_core.onnx', '', new cv.Size(320, 320), lo, 0.3, 5000); st.yuLo = lo;
      }
      return st.yu;
    }

    // mỗi hàm run* trả về box theo pixel của chính ảnh `mat` truyền vào
    function runYu(mat, lo) {
      const det = yunet(lo); det.setInputSize(new cv.Size(mat.cols, mat.rows));
      const out = new cv.Mat(); det.detect(mat, out); const res = [];
      for (let i = 0; i < out.rows; i++) { const r = out.data32F.subarray(i * 15, i * 15 + 15); res.push({ x: r[0], y: r[1], w: r[2], h: r[3], s: r[14], lm: Array.from(r.subarray(4, 14)) }); }
      out.delete(); return res;
    }
    /* --- onnxruntime-web (tuỳ chọn): opts.ort = {lib: ort, ep: 'webgpu'|'wasm'}. Lần đầu mỗi loại việc chạy cả hai bộ chạy
       và so kết quả; lệch hay lỗi thì tắt onnxruntime-web, dùng OpenCV.js (st.ortWhy ghi lý do). */
    st.useOrt = !!(opts.ort && opts.ort.lib); st.ortWhy = opts.ortWhy || null; st.ortS = new Map(); st.ver = { det: false, emb: false };
    const ortOff = why => { st.useOrt = false; st.ortWhy = why; };
    function ortSess(stem) {
      if (!st.ortS.has(stem)) {
        const raw = st.models[stem]; if (!raw) throw new Error('Thiếu model ' + stem + '.onnx');
        st.ortS.set(stem, opts.ort.lib.InferenceSession.create(raw.slice(), { executionProviders: opts.ort.ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: 'all' }));
      }
      return st.ortS.get(stem);
    }
    async function ortRun(stem, data, dims) {
      const s = await ortSess(stem), feeds = {}; feeds[s.inputNames[0]] = new opts.ort.lib.Tensor('float32', data, dims);
      const r = await s.run(feeds);
      return s.outputNames.map(n => { const t = r[n], d = t.dims; return { rows: d.length === 3 ? d[1] : d[0], cols: d.length === 3 ? d[2] : (d[1] || 1), data: t.data instanceof Float32Array ? t.data : Float32Array.from(t.data) }; });
    }
    function scPrep(bgr, sz) {
      const w = bgr.cols, h = bgr.rows, r = sz / Math.max(w, h), nw = Math.max(32, Math.round(w * r)), nh = Math.max(32, Math.round(h * r));
      const W = Math.ceil(nw / 32) * 32, H = Math.ceil(nh / 32) * 32;
      const rs = new cv.Mat(), pad = new cv.Mat(); cv.resize(bgr, rs, new cv.Size(nw, nh), 0, 0, r < 1 ? cv.INTER_AREA : cv.INTER_LINEAR);
      cv.copyMakeBorder(rs, pad, 0, H - nh, 0, W - nw, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
      const blob = cv.blobFromImage(pad, 1 / 128, new cv.Size(W, H), new cv.Scalar(127.5, 127.5, 127.5, 0), true, false); rs.delete(); pad.delete();
      return { blob, W, H, r };
    }
    function scCv(stem, P) {
      const net = netFor(stem, P.W, P.H), outs = new cv.MatVector();
      try {
        net.setInput(P.blob);
        const nm = net.getUnconnectedOutLayersNames(), names = []; for (let i = 0; i < nm.size(); i++) names.push(nm.get(i));
        net.forward2(outs, names);
        const O = []; for (let i = 0; i < outs.size(); i++) { const m = outs.get(i); O.push({ rows: m.rows, cols: m.cols, data: m.data32F.slice() }); m.delete(); }
        return O;
      } catch (e) { throw ex(e); } finally { outs.delete(); }
    }
    function scDecode(O, P, thr) {
      const { W, r } = P, Gp = { 1: [], 4: [], 10: [] }; for (const o of O) { if (!Gp[o.cols]) throw new Error('SCRFD: output lạ ' + o.rows + 'x' + o.cols); Gp[o.cols].push(o); }
      for (const c of [1, 4, 10]) { if (Gp[c].length !== 3) throw new Error('SCRFD: thiếu output'); Gp[c].sort((a, b) => b.rows - a.rows); } // theo stride 8, 16, 32
      const res = [];
      [8, 16, 32].forEach((s, i) => {
        const sc = Gp[1][i], bb = Gp[4][i], kp = Gp[10][i], gw = W / s;
        for (let n = 0; n < sc.rows; n++) {
          const v = sc.data[n]; if (v < thr) continue; const a = n >> 1, cx = (a % gw) * s, cy = Math.floor(a / gw) * s;
          const x1 = (cx - bb.data[n * 4] * s) / r, y1 = (cy - bb.data[n * 4 + 1] * s) / r, x2 = (cx + bb.data[n * 4 + 2] * s) / r, y2 = (cy + bb.data[n * 4 + 3] * s) / r;
          const lm = []; for (let j = 0; j < 5; j++) lm.push((cx + kp.data[n * 10 + 2 * j] * s) / r, (cy + kp.data[n * 10 + 2 * j + 1] * s) / r);
          res.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1, s: v, lm });
        }
      });
      return DC.nms(res, 0.4);
    }
    const sameDets = (A, B) => { // cùng các box điểm ≥ 0,4 (IoU ≥ 0,9, điểm lệch ≤ 0,05)
      const a = A.filter(d => d.s >= 0.4), b = B.filter(d => d.s >= 0.4);
      if (a.length !== b.length) return `số box ${a.length} / ${b.length}`;
      for (const d of a) if (!b.some(e => iouXY(d, e) >= 0.9 && Math.abs(d.s - e.s) <= 0.05)) return 'vị trí / điểm box lệch';
      return null;
    };
    async function runSc(stem, bgr, thr, sz) {
      const P = scPrep(bgr, sz);
      try {
        if (st.useOrt) {
          try {
            const R1 = scDecode(await ortRun(stem, P.blob.data32F.slice(), [1, 3, P.H, P.W]), P, thr);
            if (st.ver.det) return R1;
            const R0 = scDecode(scCv(stem, P), P, thr), why = sameDets(R1, R0);
            if (why) { ortOff('kết quả lệch OpenCV.js (' + why + ')'); return R0; }
            st.ver.det = true; return R1;
          } catch (e) { ortOff(String((e && e.message) || e).slice(0, 160)); }
        }
        return scDecode(scCv(stem, P), P, thr);
      } finally { P.blob.delete(); }
    }
    const place = (L, k, ox, oy) => L.map(d => ({ ...d, x: d.x / k + ox, y: d.y / k + oy, w: d.w / k, h: d.h / k, lm: d.lm.map((v, j) => v / k + (j % 2 ? oy : ox)) }));

    // cfg: {yunet, nets:[stem], lo, sz, tiles, longSide}
    async function detect(w, h, buf, cfg) {
      const rgba = cv.matFromImageData(new ImageData(new Uint8ClampedArray(buf), w, h)), bgr = new cv.Mat();
      cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
      const nDet = (cfg.yunet ? 1 : 0) + cfg.nets.length;
      const pass = async (mat, sz) => { const L = []; if (cfg.yunet) L.push(...runYu(mat, cfg.lo)); for (const s of cfg.nets) L.push(...await runSc(s, mat, cfg.lo, sz)); return nDet > 1 ? DC.fuse(L, 0.4) : L; };
      let L; const ls = Math.max(w, h);
      try {
        if (ls > cfg.longSide) { const k = cfg.longSide / ls, tmp = new cv.Mat(); cv.resize(bgr, tmp, new cv.Size(Math.round(w * k), Math.round(h * k)), 0, 0, cv.INTER_AREA); try { L = place(await pass(tmp, cfg.sz), k, 0, 0); } finally { tmp.delete(); } }
        else L = await pass(bgr, cfg.sz);
        if (cfg.tiles) {
          const tw = Math.min(w, Math.round(w * 0.6)), th = Math.min(h, Math.round(h * 0.6));
          for (const x0 of [0, w - tw]) for (const y0 of [0, h - th]) {
            const roi = bgr.roi(new cv.Rect(x0, y0, tw, th)), rs = new cv.Mat(); cv.resize(roi, rs, new cv.Size(tw * 2, th * 2), 0, 0, cv.INTER_LINEAR);
            try {
              for (const d of place(await pass(rs, Math.min(1280, Math.max(rs.cols, rs.rows))), 2, x0, y0)) {
                const cutL = x0 > 0 && d.x - x0 < 3, cutT = y0 > 0 && d.y - y0 < 3, cutR = x0 + tw < w && x0 + tw - d.x - d.w < 3, cutB = y0 + th < h && y0 + th - d.y - d.h < 3;
                if (!(cutL || cutT || cutR || cutB)) L.push(d);
              }
            } finally { roi.delete(); rs.delete(); }
          }
          L = DC.nms(L, 0.4);
        }
        // ảnh 64x64 xám của từng mặt (để luồng chính đo độ nét)
        const g64 = new cv.Mat(), gray = new cv.Mat();
        for (const d of L) {
          const x0 = Math.max(0, Math.floor(d.x)), y0 = Math.max(0, Math.floor(d.y)), x1 = Math.min(w, Math.ceil(d.x + d.w)), y1 = Math.min(h, Math.ceil(d.y + d.h));
          if (x1 - x0 < 3 || y1 - y0 < 3) continue;
          const roi = rgba.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
          cv.resize(roi, g64, new cv.Size(64, 64), 0, 0, cv.INTER_LINEAR); cv.cvtColor(g64, gray, cv.COLOR_RGBA2GRAY);
          d.g = gray.data.slice(); roi.delete();
        }
        g64.delete(); gray.delete();
      } finally { rgba.delete(); bgr.delete(); }
      return L.filter(d => d.g);
    }

    // nhúng ArcFace: lms = [[10 số], ...] toạ độ ảnh gốc -> mảng Float32Array(512) đã chuẩn hoá L2
    const l2 = e => { let n = 0; for (const v of e) n += v * v; n = Math.sqrt(n) + 1e-9; for (let i = 0; i < e.length; i++) e[i] /= n; return e; };
    function arcCv(blob) { if (!st.arc) st.arc = netFor('w600k_mbf', 112, 112, [1, 3, 112, 112]); st.arc.setInput(blob); const o = st.arc.forward(); const e = Float32Array.from(o.data32F); o.delete(); return l2(e); }
    async function embed(w, h, buf, lms) {
      const rgba = cv.matFromImageData(new ImageData(new Uint8ClampedArray(buf), w, h)), rgb = new cv.Mat(); cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB); rgba.delete();
      const res = [];
      try {
        for (const lm of lms) {
          const M = cv.matFromArray(2, 3, cv.CV_64F, DC.simMatrix(lm)), al = new cv.Mat();
          cv.warpAffine(rgb, al, M, new cv.Size(112, 112), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0)); M.delete();
          const blob = cv.blobFromImage(al, 1 / 127.5, new cv.Size(112, 112), new cv.Scalar(127.5, 127.5, 127.5, 0), false, false); al.delete();
          try {
            let e = null;
            if (st.useOrt) {
              try {
                e = l2(Float32Array.from((await ortRun('w600k_mbf', blob.data32F.slice(), [1, 3, 112, 112]))[0].data));
                if (!st.ver.emb) { const c = arcCv(blob); let d = 0; for (let i = 0; i < c.length; i++) d += c[i] * e[i]; if (e.length !== c.length || d < 0.99) { ortOff('ArcFace lệch OpenCV.js (cos ' + d.toFixed(3) + ')'); e = c; } else st.ver.emb = true; }
              } catch (er) { ortOff(String((er && er.message) || er).slice(0, 160)); e = null; }
            }
            res.push(e || arcCv(blob));
          } finally { blob.delete(); }
        }
      } catch (e) { throw ex(e); } finally { rgb.delete(); }
      return res;
    }
    const backend = () => st.useOrt ? 'onnxruntime-web (' + opts.ort.ep + ')' : 'OpenCV.js' + (st.ortWhy ? ' — không dùng onnxruntime-web: ' + st.ortWhy : '');
    return { detect, embed, backend, addModel(stem, bytes) { st.models[stem] = bytes; } };
  };
  G.DetCore = DC;
})(typeof self !== 'undefined' ? self : window);
