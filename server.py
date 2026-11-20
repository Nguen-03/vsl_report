"""Máy chủ localhost cho Dấu Tay. Chạy bằng Python 3, không cần cài thư viện."""
import json
import math
import mimetypes
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
for _import_root in (WORKSPACE_ROOT, ROOT):
    import_root = str(_import_root)
    if import_root not in __import__("sys").path:
        __import__("sys").path.insert(0, import_root)

from NgonNguKyHieu.model_adapter import predict, reload_gallery
from NgonNguKyHieu.src.add_sign import add_sign
MAX_BYTES = 100 * 1024 * 1024
STATIC = {"/": "index.html", "/index.html": "index.html", "/style.css": "style.css",
          "/demo-layout.css": "demo-layout.css", "/app.js": "app.js",
          "/assets/hoang-hon.png": "assets/hoang-hon.png"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
ADAPTER = predict
ADAPTER_ERROR = None
MODEL_LOCK = threading.Lock()


def load_adapter():
    return

class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, payload):
        self.send_data(status, "application/json; charset=utf-8-sig",
                       json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/health":
            mode = "error" if ADAPTER_ERROR is not None else "model" if ADAPTER else "demo"
            self.send_json(200, {"mode": mode})
            return
        if path not in STATIC:
            self.send_json(404, {"error": "Không tìm thấy trang."})
            return
        asset = ROOT / STATIC[path]
        content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        if asset.suffix in {".html", ".css", ".js"}:
            content_type += "; charset=utf-8"
        self.send_data(200, content_type, asset.read_bytes())

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path not in {"/api/predict", "/api/add-sign"}:
            self.send_json(404, {"error": "Không tìm thấy API."})
            return
        is_add_sign = url.path == "/api/add-sign"
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}",
                                      f"http://localhost:{self.server.server_port}"}:
            self.send_json(403, {"error": "Hãy gửi video từ giao diện localhost."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if size <= 0 or size > MAX_BYTES:
            self.send_json(413, {"error": "Video trống hoặc lớn hơn 100 MB."})
            return
        if ADAPTER_ERROR is not None:
            self.send_json(503, {"error": "Không nạp được model_adapter. Xem lỗi trong cửa sổ Python."})
            return
        if is_add_sign and ADAPTER is None:
            self.send_json(503, {"error": "Cần kết nối model để thêm ký hiệu."})
            return

        query = parse_qs(url.query)
        label = query.get("label", [""])[0].strip() if is_add_sign else None
        if is_add_sign:
            if not label or len(label) > 80 or any(ord(char) < 32 for char in label):
                self.send_json(400, {"error": "Tên ký hiệu không hợp lệ."})
                return
        else:
            source = query.get("source", ["video"])[0]
            if source not in {"video", "camera"}:
                self.send_json(400, {"error": "Nguồn video không hợp lệ."})
                return
        extension = Path(query.get("name", ["clip"])[0]).suffix.lower()
        content_type = self.headers.get_content_type()
        if extension not in VIDEO_EXTENSIONS:
            extension = mimetypes.guess_extension(content_type) or ".video"
        try:
            self.connection.settimeout(30)
            with tempfile.TemporaryDirectory(prefix="dau-tay-") as folder:
                path = Path(folder) / ("clip" + extension)
                with path.open("wb") as target:
                    remaining = size
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError("Video gửi lên chưa đầy đủ.")
                        target.write(chunk)
                        remaining -= len(chunk)
                if is_add_sign:
                    with MODEL_LOCK:
                        saved_label = add_sign(path, label)
                        reload_gallery()
                    payload = {"label": saved_label}
                elif ADAPTER is None:
                    payload = {"label": "Xin chào", "cosine_distance": None, "top5": [], "demo": True}
                else:
                    with MODEL_LOCK:
                        result = ADAPTER(path, source=source)
                    label = result["label"]
                    if not isinstance(label, str) or not label.strip():
                        raise ValueError("Adapter phải trả label dạng chuỗi khác rỗng.")
                    cosine_distance = result.get("cosine_distance")
                    top5 = result.get("top5")
                    if not isinstance(top5, list):
                        raise ValueError("top5 phải là danh sách kết quả.")
                    normalized_top5 = []
                    for rank in top5[:5]:
                        if not isinstance(rank, dict):
                            raise ValueError("Mỗi kết quả top5 phải là object.")
                        rank_label = rank.get("label")
                        rank_distance = rank.get("cosine_distance")
                        if not isinstance(rank_label, str) or not rank_label.strip():
                            raise ValueError("Nhãn trong top5 không hợp lệ.")
                        rank_distance = float(rank_distance)
                        if not math.isfinite(rank_distance) or not 0 <= rank_distance <= 2:
                            raise ValueError("Khoảng cách cosine trong top5 phải nằm trong [0, 2].")
                        normalized_top5.append({
                            "label": rank_label.strip(),
                            "cosine_distance": rank_distance,
                        })
                    normalized_top5.sort(key=lambda rank: rank["cosine_distance"])
                    if not normalized_top5:
                        raise ValueError("Adapter không trả về kết quả top5.")
                    if cosine_distance is None:
                        cosine_distance = normalized_top5[0]["cosine_distance"]
                    cosine_distance = float(cosine_distance)
                    if not math.isfinite(cosine_distance) or not 0 <= cosine_distance <= 2:
                        raise ValueError("Khoảng cách cosine phải nằm trong [0, 2].")
                    payload = {
                        "label": label.strip(),
                        "cosine_distance": cosine_distance,
                        "top5": normalized_top5,
                        "demo": False,
                    }
            self.send_json(200, payload)
        except TimeoutError:
            self.send_json(408, {"error": "Quá thời gian đọc video. Hãy gửi lại."})
        except Exception as error:
            print("Processing error:", error)
            self.send_json(500, {"error": "Không xử lý được video. Xem lỗi trong cửa sổ Python."})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    load_adapter()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Server is running at {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
