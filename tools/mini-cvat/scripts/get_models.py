#!/usr/bin/env python3
"""Tải model cho "Tự gán nhãn" của Mini CVAT vào thư mục models/ (chỉ dùng thư viện chuẩn của Python).

  python scripts/get_models.py                 # det_500m + w600k_mbf (~16 MB)
  python scripts/get_models.py --all           # thêm det_10g (~17 MB, SCRFD chính xác hơn)
  python scripts/get_models.py det_10g         # chỉ model chỉ định
  python scripts/get_models.py --ort           # gói onnxruntime-web (npm) vào lib/ort/, để chạy model nhanh hơn

Model lấy từ gói insightface "buffalo_sc" / "buffalo_l" (GitHub releases v0.7). Chỉ đọc đúng file cần trong zip
(HTTP Range), không phải tải cả gói 288 MB. Kiểm tra SHA-256 sau khi tải.
LƯU Ý GIẤY PHÉP: các model insightface pretrained chỉ cho nghiên cứu phi thương mại. Dùng để gán nhãn/đánh giá nội bộ thì ổn;
nếu đưa vào sản phẩm thương mại phải dùng model có giấy phép phù hợp."""
import hashlib, os, struct, sys, urllib.request, zlib

BASE = 'https://github.com/deepinsight/insightface/releases/download/v0.7/'
MODELS = {  # stem -> (zip, tên file trong zip, sha256)
    'det_500m': ('buffalo_sc.zip', 'det_500m.onnx', '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a'),
    'w600k_mbf': ('buffalo_sc.zip', 'w600k_mbf.onnx', '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f'),
    'det_10g': ('buffalo_l.zip', 'det_10g.onnx', '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'),
}
DEFAULT = ['det_500m', 'w600k_mbf']
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.path.dirname(HERE), 'models')


def _get(url, rng=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'mini-cvat'})
    if rng:
        req.add_header('Range', 'bytes=%s' % rng)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read(), r.headers


def _size(url):
    st, _, h = _get(url, '0-0')
    cr = h.get('Content-Range')
    return int(cr.split('/')[-1]) if cr else None


def _remote_member(url, member):
    """Đọc một file trong zip từ xa bằng HTTP Range. Trả None nếu server không hỗ trợ Range."""
    n = _size(url)
    if not n:
        return None
    tail = min(n, 1 << 20)
    _, t, _ = _get(url, '%d-%d' % (n - tail, n - 1))
    e = t.rfind(b'PK\x05\x06')
    if e < 0:
        return None
    cnt, cd_size, cd_off = struct.unpack('<HII', t[e + 10:e + 20])
    if cd_off == 0xFFFFFFFF:
        return None  # zip64: dùng đường tải đầy đủ
    cd = t[cd_off - (n - tail):] if cd_off >= n - tail else _get(url, '%d-%d' % (cd_off, cd_off + cd_size - 1))[1]
    p = 0
    while p < len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        meth, = struct.unpack('<H', cd[p + 10:p + 12])
        csize, usize = struct.unpack('<II', cd[p + 20:p + 28])
        nl, el, cl = struct.unpack('<HHH', cd[p + 28:p + 34])
        lho, = struct.unpack('<I', cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nl].decode('utf-8', 'replace')
        if name == member:
            lh = _get(url, '%d-%d' % (lho, lho + 29))[1]
            lnl, lel = struct.unpack('<HH', lh[26:30])
            start = lho + 30 + lnl + lel
            data = _get(url, '%d-%d' % (start, start + csize - 1))[1]
            return data if meth == 0 else zlib.decompress(data, -15)
        p += 46 + nl + el + cl
    return None


def _full_member(url, member):
    import io, zipfile
    _, data, _ = _get(url)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(member)


def ensure(stem, log=print):
    """Bảo đảm models/<stem>.onnx tồn tại và đúng SHA-256. Trả về đường dẫn file."""
    if stem not in MODELS:
        raise ValueError('Model không biết: %s (có: %s)' % (stem, ', '.join(MODELS)))
    zipname, member, sha = MODELS[stem]
    path = os.path.join(DEST, stem + '.onnx')
    if os.path.isfile(path) and hashlib.sha256(open(path, 'rb').read()).hexdigest() == sha:
        return path
    os.makedirs(DEST, exist_ok=True)
    url = BASE + zipname
    log('Đang tải %s từ %s ...' % (stem, zipname))
    data = None
    try:
        data = _remote_member(url, member)
    except Exception as e:  # noqa
        log('Tải từng phần không được (%s), thử tải cả gói' % e)
    if data is None:
        data = _full_member(url, member)
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError('SHA-256 của %s không khớp, file có thể hỏng' % stem)
    tmp = path + '.part'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)
    log('Đã lưu %s (%.1f MB)' % (path, len(data) / 1e6))
    return path


ORT_VERSION = '1.20.1'
ORT_DIR = os.path.join(os.path.dirname(HERE), 'lib', 'ort')
ORT_MAIN = ['ort.all.min.js', 'ort.webgpu.min.js', 'ort.min.js']


def ensure_ort(log=print):
    """Tải onnxruntime-web (npm, MIT) vào lib/ort/ và ghi lib/ort/manifest.json. Kiểm tra mã băm npm công bố (sha512)."""
    import base64, io, json, tarfile
    man = os.path.join(ORT_DIR, 'manifest.json')
    if os.path.isfile(man):
        try:
            m = json.load(open(man))
            if m.get('version') == ORT_VERSION and os.path.isfile(os.path.join(ORT_DIR, m['main'])):
                return man
        except Exception:  # noqa
            pass
    log('Đang tải onnxruntime-web %s từ npm ...' % ORT_VERSION)
    meta = json.loads(_get('https://registry.npmjs.org/onnxruntime-web/' + ORT_VERSION)[1])
    tgz = _get(meta['dist']['tarball'])[1]
    integ = meta['dist'].get('integrity', '')
    if integ.startswith('sha512-'):
        if base64.b64encode(hashlib.sha512(tgz).digest()).decode() != integ[7:]:
            raise RuntimeError('Mã băm gói onnxruntime-web không khớp')
    elif hashlib.sha1(tgz).hexdigest() != meta['dist'].get('shasum'):
        raise RuntimeError('Mã băm gói onnxruntime-web không khớp')
    os.makedirs(ORT_DIR, exist_ok=True)
    names = []
    with tarfile.open(fileobj=io.BytesIO(tgz), mode='r:gz') as t:
        for mbr in t.getmembers():
            n = mbr.name
            if not mbr.isfile() or not n.startswith('package/dist/') or '/' in n[len('package/dist/'):]:
                continue
            base = n[len('package/dist/'):]
            if base.endswith(('.map', '.d.ts')) or not base.endswith(('.js', '.mjs', '.wasm')):
                continue
            with open(os.path.join(ORT_DIR, base), 'wb') as f:
                f.write(t.extractfile(mbr).read())
            names.append(base)
    main = next((x for x in ORT_MAIN if x in names), None)
    if not main:
        raise RuntimeError('Gói onnxruntime-web không có ' + ' / '.join(ORT_MAIN))
    json.dump({'version': ORT_VERSION, 'main': main, 'files': sorted(names)}, open(man, 'w'))
    log('Đã lưu onnxruntime-web vào %s (%d file)' % (ORT_DIR, len(names)))
    return man


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('-')]
    names = list(MODELS) if '--all' in sys.argv else (a or ([] if '--ort' in sys.argv else DEFAULT))
    for s in names:
        ensure(s)
    if '--ort' in sys.argv:
        ensure_ort()
    print('Xong. Mở Mini CVAT bằng: python scripts/serve.py')
