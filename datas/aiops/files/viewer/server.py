"""数据预览网站服务（零第三方依赖）。

用法：
    python datas/aiops/files/viewer/server.py [--host 0.0.0.0] [--port 8000]

- 自动扫描 files/jsons 下的 JSON 文件作为数据集；
- /api/items 支持按 curve / recover / source / 名称筛选与分页；
- /api/image 按记录的 from 字段从 gen / labeled 目录读取图像。
"""

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FILES_ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = FILES_ROOT / "jsons"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# from 字段 -> 图像目录；新增来源时只需在此登记
IMAGE_DIRS = {
    "gen": FILES_ROOT / "gen",
    "labeled": FILES_ROOT / "labeled",
}

DEFAULT_LIMIT = 60
MAX_LIMIT = 200

_cache = {}


def load_dataset(name):
    """加载并缓存一个 JSON 数据集，只接受 jsons 目录下的文件。"""
    if name not in _cache:
        if not re.fullmatch(r"[\w.-]+\.json", name):
            raise ValueError("非法数据集名称")
        path = JSON_DIR / name
        if not path.is_file():
            raise FileNotFoundError(name)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("数据集必须是记录列表")
        _cache[name] = data
    return _cache[name]


def list_datasets():
    datasets = []
    for path in sorted(JSON_DIR.glob("*.json")):
        try:
            data = load_dataset(path.name)
        except (ValueError, json.JSONDecodeError):
            continue
        curves = sorted({r.get("curve", "other") for r in data})
        recovers = sorted({r.get("recover", "other") for r in data})
        sources = sorted({resolve_source(r) for r in data})
        datasets.append({
            "name": path.name,
            "count": len(data),
            "curves": curves,
            "recovers": recovers,
            "sources": sources,
        })
    return datasets


def resolve_source(record):
    """确定图像所在来源目录；from 缺失或目录中不存在时回退探测。"""
    image = Path(record.get("image", "")).name
    source = record.get("from")
    if source in IMAGE_DIRS and (IMAGE_DIRS[source] / image).is_file():
        return source
    for name, directory in IMAGE_DIRS.items():
        if (directory / image).is_file():
            return name
    return source if source in IMAGE_DIRS else next(iter(IMAGE_DIRS))


def query_items(params):
    dataset = params.get("dataset", "")
    data = load_dataset(dataset)

    curve = params.get("curve", "")
    recover = params.get("recover", "")
    source = params.get("source", "")
    keyword = params.get("q", "").lower()

    filtered = []
    for record in data:
        if curve and record.get("curve") != curve:
            continue
        if recover and record.get("recover") != recover:
            continue
        src = resolve_source(record)
        if source and src != source:
            continue
        if keyword and keyword not in record.get("image", "").lower():
            continue
        filtered.append((record, src))

    total = len(filtered)
    try:
        offset = max(int(params.get("offset", 0)), 0)
        limit = min(max(int(params.get("limit", DEFAULT_LIMIT)), 1), MAX_LIMIT)
    except ValueError:
        offset, limit = 0, DEFAULT_LIMIT

    page = filtered[offset:offset + limit]
    items = []
    for index, (record, src) in enumerate(page, start=offset + 1):
        items.append({
            "index": index,
            "curve": record.get("curve", "other"),
            "recover": record.get("recover", "other"),
            "source": src,
            "image": record.get("image", ""),
            "image_url": f"/api/image?source={src}&name={record.get('image', '')}",
        })
    return {"total": total, "offset": offset, "limit": limit, "items": items}


def image_path(params):
    source = params.get("source", "")
    name = Path(params.get("name", "")).name  # 去除任何路径分量，防止目录穿越
    if source not in IMAGE_DIRS or not name:
        return None
    path = (IMAGE_DIRS[source] / name).resolve()
    base = IMAGE_DIRS[source].resolve()
    return path if base in path.parents and path.is_file() else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 精简日志
        print(f"[{self.command}] {self.path}")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        try:
            if path == "/api/datasets":
                self._send_json(list_datasets())
            elif path == "/api/items":
                self._send_json(query_items(params))
            elif path == "/api/image":
                target = image_path(params)
                if target is None:
                    self._send_json({"error": "图像不存在"}, 404)
                else:
                    self._send_file(target)
            elif path == "/":
                self._send_file(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                target = (STATIC_DIR / path[len("/static/"):]).resolve()
                if STATIC_DIR.resolve() in target.parents and target.is_file():
                    self._send_file(target)
                else:
                    self._send_json({"error": "not found"}, 404)
            else:
                self._send_json({"error": "not found"}, 404)
        except (ValueError, FileNotFoundError) as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"服务器内部错误: {e}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="数据预览网站")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"数据预览服务已启动: http://{args.host}:{args.port}")
    print(f"数据集目录: {JSON_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
