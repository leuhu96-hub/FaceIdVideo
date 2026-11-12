/* Đọc frame tuần tự bằng WebCodecs (không nhảy từng frame bằng <video>.currentTime).
   Tự tách gói MP4 (H.264 avc1/avc3, HEVC hvc1/hev1; có góc xoay của video điện thoại) và WebM/MKV (VP8, VP9, H.264).
   Frame trả về cho frame số f đúng như <video> hiển thị khi đặt currentTime=(f+0.5)/fps: frame cuối có thời điểm ≤ thời điểm đó.
   Không hỗ trợ (codec khác, file lạ, trình duyệt không có VideoDecoder) thì ném lỗi để nơi gọi quay về cách nhảy frame cũ.
   API: SeqReader.open(file, fps) -> {rot, codec, frames(targets) : async generator {f, frame(VideoFrame)}, close()}
        SeqReader.draw(ctx, frame, W, H, rot) vẽ frame (đã xoay) lên canvas W x H */
(function (G) {
  'use strict';
  const SR = {};

  class Rd {
    constructor(file) { this.f = file; this.size = file.size; this.buf = null; this.base = 0; }
    async get(pos, len) {
      if (!this.buf || pos < this.base || pos + len > this.base + this.buf.length) {
        const end = Math.min(this.size, pos + Math.max(len, 4 << 20));
        this.buf = new Uint8Array(await this.f.slice(pos, end).arrayBuffer()); this.base = pos;
      }
      return this.buf.subarray(pos - this.base, pos - this.base + len);
    }
  }
  const u32 = (b, i) => ((b[i] << 24) >>> 0) + (b[i + 1] << 16) + (b[i + 2] << 8) + b[i + 3];
  const s32 = (b, i) => (b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3];
  const u64 = (b, i) => u32(b, i) * 4294967296 + u32(b, i + 4);
  const str4 = (b, i) => String.fromCharCode(b[i], b[i + 1], b[i + 2], b[i + 3]);
  const hex2 = v => v.toString(16).padStart(2, '0');

  /* ---------------- MP4 ---------------- */
  function boxes(b, s, e) {
    const out = []; let i = s;
    while (i + 8 <= e) {
      let sz = u32(b, i), h = 8; const t = str4(b, i + 4);
      if (sz === 1) { sz = u64(b, i + 8); h = 16; } else if (sz === 0) sz = e - i;
      if (sz < h) break; out.push({ t, s: i + h, e: Math.min(e, i + sz) }); i += sz;
    }
    return out;
  }
  const child = (b, x, t) => x && boxes(b, x.s, x.e).find(y => y.t === t);
  function hevcCodec(c) { // chuỗi codec từ hvcC (giống mp4box.js)
    const sp = c[1] >> 6, tier = (c[1] >> 5) & 1, prof = c[1] & 31;
    let comp = u32(c, 2), rev = 0; for (let i = 0; i < 32; i++) { rev = (rev << 1) | (comp & 1); comp >>>= 1; }
    const cons = Array.from(c.subarray(6, 12)); while (cons.length && cons[cons.length - 1] === 0) cons.pop();
    return 'hvc1.' + (['', 'A', 'B', 'C'][sp]) + prof + '.' + (rev >>> 0).toString(16).toUpperCase() + '.' + (tier ? 'H' : 'L') + c[12] + (cons.length ? '.' + cons.map(x => x.toString(16).toUpperCase()).join('.') : '');
  }
  async function mp4(rd) {
    let pos = 0, moov = null, hl = 8;
    while (pos + 8 <= rd.size) {
      const h = await rd.get(pos, 16); let sz = u32(h, 0); const t = str4(h, 4); hl = 8;
      if (sz === 1) { sz = u64(h, 8); hl = 16; } else if (sz === 0) sz = rd.size - pos;
      if (sz < 8) throw new Error('MP4 hỏng');
      if (t === 'moov') { moov = new Uint8Array(await rd.f.slice(pos, pos + sz).arrayBuffer()); break; }
      pos += sz;
    }
    if (!moov) throw new Error('MP4: không thấy moov');
    const b = moov;
    for (const trak of boxes(b, hl, b.length).filter(x => x.t === 'trak')) {
      const mdia = child(b, trak, 'mdia'), hdlr = child(b, mdia, 'hdlr');
      if (!hdlr || str4(b, hdlr.s + 8) !== 'vide') continue;
      const tk = child(b, trak, 'tkhd'), m = tk.s + (b[tk.s] === 1 ? 52 : 40);
      const rot = ((Math.round(Math.atan2(s32(b, m + 4) / 65536, s32(b, m) / 65536) * 180 / Math.PI) % 360) + 360) % 360;
      const md = child(b, mdia, 'mdhd'), ts = u32(b, md.s + (b[md.s] === 1 ? 20 : 12));
      const stbl = child(b, child(b, mdia, 'minf'), 'stbl');
      const stsd = child(b, stbl, 'stsd'), ent = boxes(b, stsd.s + 8, stsd.e)[0], typ = ent.t;
      const sub = boxes(b, ent.s + 78, ent.e);
      let codec, desc;
      if (typ === 'avc1' || typ === 'avc3') { const c = sub.find(x => x.t === 'avcC'); if (!c) throw new Error('MP4: thiếu avcC'); desc = b.slice(c.s, c.e); codec = typ + '.' + hex2(desc[1]) + hex2(desc[2]) + hex2(desc[3]); }
      else if (typ === 'hvc1' || typ === 'hev1') { const c = sub.find(x => x.t === 'hvcC'); if (!c) throw new Error('MP4: thiếu hvcC'); desc = b.slice(c.s, c.e); codec = hevcCodec(desc).replace(/^hvc1/, typ); }
      else throw new Error('Codec MP4 chưa hỗ trợ: ' + typ);
      // bảng mẫu
      const tab = n => child(b, stbl, n);
      const stts = tab('stts'), ctts = tab('ctts'), stss = tab('stss'), stsz = tab('stsz'), stsc = tab('stsc'), stco = tab('stco') || tab('co64'), co64 = !tab('stco');
      const N = u32(b, stsz.s + 8), fixed = u32(b, stsz.s + 4), size = i => fixed || u32(b, stsz.s + 12 + 4 * i);
      const S = new Array(N);
      { let i = 0, d = 0; const n = u32(b, stts.s + 4); for (let k = 0; k < n; k++) { const c = u32(b, stts.s + 8 + 8 * k), dl = u32(b, stts.s + 12 + 8 * k); for (let j = 0; j < c && i < N; j++, i++) { S[i] = { dts: d, cts: d, key: !stss, size: size(i) }; d += dl; } } }
      if (ctts) { let i = 0; const n = u32(b, ctts.s + 4); for (let k = 0; k < n; k++) { const c = u32(b, ctts.s + 8 + 8 * k), o = s32(b, ctts.s + 12 + 8 * k); for (let j = 0; j < c && i < N; j++, i++) S[i].cts += o; } }
      if (stss) { const n = u32(b, stss.s + 4); for (let k = 0; k < n; k++) { const i = u32(b, stss.s + 8 + 4 * k) - 1; if (S[i]) S[i].key = true; } }
      { const nc = u32(b, stco.s + 4), off = c => co64 ? u64(b, stco.s + 8 + 8 * c) : u32(b, stco.s + 8 + 4 * c);
        const ne = u32(b, stsc.s + 4), E = []; for (let k = 0; k < ne; k++) E.push([u32(b, stsc.s + 8 + 12 * k) - 1, u32(b, stsc.s + 12 + 12 * k)]);
        let i = 0; for (let c = 0; c < nc && i < N; c++) { let e = 0; while (e + 1 < E.length && E[e + 1][0] <= c) e++; let o = off(c); for (let j = 0; j < E[e][1] && i < N; j++, i++) { S[i].off = o; o += S[i].size; } } }
      const t0 = Math.min(...S.map(x => x.cts));
      for (const x of S) x.pts = Math.round((x.cts - t0) / ts * 1e6);
      return { codec, desc, rot, samples: S };
    }
    throw new Error('MP4: không có track video');
  }

  /* ---------------- WebM / Matroska ---------------- */
  function vint(b, i, strip) {
    const f = b[i]; let l = 1, m = 0x80; while (l <= 8 && !(f & m)) { l++; m >>= 1; }
    if (l > 8) throw new Error('EBML hỏng');
    let v = strip ? f & (m - 1) : f, allOnes = strip && (f & (m - 1)) === m - 1;
    for (let k = 1; k < l; k++) { v = v * 256 + b[i + k]; if (b[i + k] !== 255) allOnes = false; }
    return [v, l, allOnes];
  }
  const LVL1 = new Set([0x1F43B675, 0x1C53BB6B, 0x1254C367, 0x1043A770, 0x1941A469, 0x114D9B74, 0x1549A966, 0x1654AE6B]);
  async function webm(rd) {
    let pos = 0, end = rd.size, scale = 1e6, vtrack = null, codecId = '', priv = null, cluster = 0;
    const S = [];
    const uint = (b, s, n) => { let v = 0; for (let k = 0; k < n; k++) v = v * 256 + b[s + k]; return v; };
    const readEl = async p => { const h = await rd.get(p, 16); const [id, il] = vint(h, 0, false), [sz, sl, unk] = vint(h, il, true); return { id, s: p + il + sl, sz: unk ? null : sz }; };
    const block = (b, simple, groupRef) => {
      const [tn, tl] = vint(b, 0, true); if (tn !== vtrack) return null;
      const rel = (b[tl] << 8 | b[tl + 1]) << 16 >> 16, fl = b[tl + 2];
      if (fl & 0x06) throw new Error('WebM: block có lacing, chưa hỗ trợ');
      return { hdr: tl + 3, pts: (cluster + rel) * scale / 1000, key: simple ? !!(fl & 0x80) : !groupRef };
    };
    while (pos < end) {
      const e = await readEl(pos); const id = e.id;
      if (id === 0x18538067) { pos = e.s; if (e.sz !== null) end = Math.min(rd.size, e.s + e.sz); continue; } // Segment: đi vào trong
      if (id === 0x1F43B675) { pos = e.s; continue; } // Cluster: đi vào trong (các phần tử con xử lý tuần tự)
      if (e.sz === null) { pos = e.s; continue; }
      if (id === 0x1549A966 || id === 0x1654AE6B) { // Info, Tracks: đọc hết
        const b = (await rd.get(e.s, e.sz)).slice();
        const walk = (s, t, fn) => { let i = s; while (i < t) { const [cid, il] = vint(b, i, false), [csz, sl] = vint(b, i + il, true); fn(cid, i + il + sl, csz); i += il + sl + csz; } };
        if (id === 0x1549A966) walk(0, b.length, (cid, s, n) => { if (cid === 0x2AD7B1) scale = uint(b, s, n); });
        else walk(0, b.length, (cid, s, n) => { if (cid !== 0xAE || vtrack !== null) return; let num = 0, type = 0, cod = '', pr = null;
          walk(s, s + n, (k, s2, n2) => { if (k === 0xD7) num = uint(b, s2, n2); else if (k === 0x83) type = uint(b, s2, n2); else if (k === 0x86) cod = String.fromCharCode(...b.subarray(s2, s2 + n2)); else if (k === 0x63A2) pr = b.slice(s2, s2 + n2); });
          if (type === 1) { vtrack = num; codecId = cod; priv = pr; } });
      } else if (id === 0xE7) { cluster = uint(await rd.get(e.s, e.sz), 0, e.sz); }
      else if (id === 0xA3 && vtrack !== null) { const h = await rd.get(e.s, Math.min(e.sz, 16)), x = block(h, true); if (x) S.push({ off: e.s + x.hdr, size: e.sz - x.hdr, pts: x.pts, key: x.key }); }
      else if (id === 0xA0 && vtrack !== null) { // BlockGroup
        const b = (await rd.get(e.s, e.sz)).slice(); let i = 0, blk = null, ref = false;
        while (i < b.length) { const [cid, il] = vint(b, i, false), [csz, sl] = vint(b, i + il, true), s = i + il + sl; if (cid === 0xA1) blk = { s, n: csz }; else if (cid === 0xFB) ref = true; i = s + csz; }
        if (blk) { const x = block(b.subarray(blk.s, blk.s + Math.min(blk.n, 16)), false, ref); if (x) S.push({ off: e.s + blk.s + x.hdr, size: blk.n - x.hdr, pts: x.pts, key: x.key }); }
      }
      pos = e.s + e.sz;
    }
    if (vtrack === null) throw new Error('WebM: không có track video');
    let codec, desc;
    if (codecId === 'V_VP8') codec = 'vp8';
    else if (codecId === 'V_VP9') codec = 'vp09.00.10.08';
    else if (codecId === 'V_MPEG4/ISO/AVC' && priv) { desc = priv; codec = 'avc1.' + hex2(priv[1]) + hex2(priv[2]) + hex2(priv[3]); }
    else throw new Error('Codec WebM chưa hỗ trợ: ' + codecId);
    const t0 = Math.min(...S.map(x => x.pts)); for (const x of S) x.pts = Math.round(x.pts - t0);
    return { codec, desc, rot: 0, samples: S };
  }

  SR.supported = () => typeof G.VideoDecoder === 'function' && typeof G.EncodedVideoChunk === 'function';
  SR.open = async function (file, fps) {
    if (!SR.supported()) throw new Error('Trình duyệt không có WebCodecs');
    const rd = new Rd(file), head = await rd.get(0, 12);
    const dm = str4(head, 4) === 'ftyp' ? await mp4(rd) : u32(head, 0) === 0x1A45DFA3 ? await webm(rd) : null;
    if (!dm) throw new Error('Chỉ hỗ trợ MP4 / WebM');
    const S = dm.samples; if (!S.length) throw new Error('Video không có frame');
    const cfg = { codec: dm.codec, optimizeForLatency: false };
    if (dm.desc) cfg.description = dm.desc;
    const sup = await VideoDecoder.isConfigSupported(cfg); if (!sup.supported) throw new Error('Trình duyệt không giải mã được ' + dm.codec);
    const keys = []; S.forEach((x, i) => { if (x.key) keys.push(i); });
    if (!keys.length) throw new Error('Video không có keyframe');
    const readers = [];
    async function* frames(targets) {
      const T = targets.length, tgt = i => (targets[i] + 0.5) / fps * 1e6;
      const q = []; let err = null, done = false, wake = null, si = keys[0], prev = null, ti = 0, flushing = false, ended = false, cur = null;
      const poke = () => { if (wake) { const w = wake; wake = null; w(); } };
      const dec = new VideoDecoder({ output: fr => { if (ended) { fr.close(); return; } q.push(fr); poke(); }, error: e => { err = e; poke(); } });
      readers.push(dec); dec.configure(cfg);
      const feed = async () => {
        while (si < S.length && dec.decodeQueueSize < 12 && q.length < 6 && ti < T) {
          // nhảy tới keyframe gần nhất trước frame cần đọc nếu còn ở xa (frame cần đọc thưa)
          const tt = tgt(ti); let lo = 0, hi = keys.length - 1, k = -1;
          while (lo <= hi) { const m = (lo + hi) >> 1; if (S[keys[m]].pts <= tt) { k = keys[m]; lo = m + 1; } else hi = m - 1; }
          if (k > si) si = k;
          const x = S[si], data = (await rd.get(x.off, x.size)).slice();
          dec.decode(new EncodedVideoChunk({ type: x.key ? 'key' : 'delta', timestamp: x.pts, data })); si++;
        }
        if ((si >= S.length || ti >= T) && !flushing) { flushing = true; dec.flush().then(() => { done = true; poke(); }, e => { if (!err) err = e; poke(); }); }
      };
      try {
        while (ti < T) {
          if (err) throw new Error('Lỗi giải mã: ' + (err.message || err));
          if (q.length) {
            const fr = cur = q.shift();
            while (ti < T && fr.timestamp > tgt(ti)) { yield { f: targets[ti], frame: prev || fr }; ti++; }
            if (prev) prev.close(); prev = fr; cur = null; continue;
          }
          if (done) { while (ti < T) { if (!prev) throw new Error('Không giải mã được frame nào'); yield { f: targets[ti], frame: prev }; ti++; } break; }
          await feed();
          if (!q.length && !done && !err) await new Promise(r => { wake = r; if (q.length || done || err) poke(); });
        }
      } finally {
        ended = true; if (cur && cur !== prev) cur.close(); if (prev) prev.close(); q.forEach(f => f.close()); try { if (dec.state !== 'closed') dec.close(); } catch (e) { }
      }
    }
    return { rot: dm.rot, codec: dm.codec, n: S.length, frames, close() { readers.forEach(d => { try { if (d.state !== 'closed') d.close(); } catch (e) { } }); } };
  };
  SR.draw = function (ctx, fr, W, H, rot) {
    if (!rot) { ctx.drawImage(fr, 0, 0, W, H); return; }
    ctx.save();
    if (rot === 90) { ctx.translate(W, 0); ctx.rotate(Math.PI / 2); ctx.drawImage(fr, 0, 0, H, W); }
    else if (rot === 180) { ctx.translate(W, H); ctx.rotate(Math.PI); ctx.drawImage(fr, 0, 0, W, H); }
    else { ctx.translate(0, H); ctx.rotate(-Math.PI / 2); ctx.drawImage(fr, 0, 0, H, W); }
    ctx.restore();
  };
  G.SeqReader = SR;
})(typeof self !== 'undefined' ? self : window);
