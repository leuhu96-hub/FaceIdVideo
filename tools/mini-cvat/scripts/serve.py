#!/usr/bin/env python3
"""Web server cho Mini CVAT (thay cho `python -m http.server`).
Khác bản thường: khi "Tự gán nhãn" cần model SCRFD / ArcFace mà chưa có trong models/, trình duyệt gọi
/api/ensure-model và server tải model về (một lần). Chỉ dùng thư viện chuẩn.

  python scripts/serve.py            # http://127.0.0.1:8000
  python scripts/serve.py --port 9000"""
import json, os, sys, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import get_models  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = threading.Lock()


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/api/ensure-model':
            name = (parse_qs(u.query).get('name') or [''])[0]
            try:
                with LOCK:
                    get_models.ensure(name, log=lambda m: print('[models]', m, flush=True))
                out, code = {'ok': True}, 200
            except Exception as e:  # noqa
                print('[models] lỗi:', e, flush=True)
                out, code = {'ok': False, 'error': 'Không tải được model %s: %s. Kiểm tra mạng hoặc chạy python scripts/get_models.py' % (name, e)}, 200
            b = json.dumps(out).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        return super().do_GET()

    def log_message(self, fmt, *a):
        if '/api/' in (a[0] if a else ''):
            super().log_message(fmt, *a)


if __name__ == '__main__':
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8000
    print('Mini CVAT: http://127.0.0.1:%d   (Ctrl+C để dừng)' % port)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
