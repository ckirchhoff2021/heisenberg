"""JEV FastAPI service integration example.

Start the service first:

    .venv/bin/python infer/jev_server.py \
        --checkpoint outputs/jev/best.pt \
        --port 8008

Then run one of:

    .venv/bin/python tests/test_jev_server.py
    .venv/bin/python tests/test_jev_server.py --mode path
    .venv/bin/python tests/test_jev_server.py --mode file --sample-index 3
    .venv/bin/python tests/test_jev_server.py --mode all --assert-labels

The default ``base64`` mode also works when the client and server do not share
the same filesystem. ``path`` mode requires the image path to exist on the
server host.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid


DEFAULT_DATA_FILE = "datas/aiops/files/jsons/argos-valids.json"
DEFAULT_FILES_ROOT = "datas/aiops/files"
CURVE_LABELS = {"normal", "offset", "increase", "decline", "spike", "fluctuate", "other"}
RECOVER_LABELS = {"recovered", "recovering", "no_recover", "other"}


def load_sample(data_file: str, files_root: str, index: int) -> dict:
    with open(data_file, encoding="utf-8") as f:
        records = json.load(f)
    if not 0 <= index < len(records):
        raise IndexError(f"sample index {index} is outside [0, {len(records) - 1}]")

    record = records[index]
    image_path = os.path.abspath(os.path.join(files_root, record["from"], record["image"]))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    return {
        "index": index,
        "image_path": image_path,
        "curve": record["curve"],
        "recover": record["recover"],
    }


def request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 120.0,
) -> tuple[dict, float]:
    headers = {"Content-Type": content_type} if content_type else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach JEV server at {url}: {exc.reason}") from exc
    return payload, (time.perf_counter() - started) * 1000


def request_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    return request(
        url,
        method="POST",
        body=json.dumps(payload).encode("utf-8"),
        content_type="application/json",
        timeout=timeout,
    )


def request_file(url: str, image_path: str, timeout: float) -> tuple[dict, float]:
    boundary = f"jev-{uuid.uuid4().hex}"
    filename = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()
    return request(
        url,
        method="POST",
        body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
        timeout=timeout,
    )


def validate_prediction(result: dict) -> None:
    if set(result) != {"trend", "recover"}:
        raise AssertionError(f"unexpected response keys: {sorted(result)}")

    for name, labels in (("trend", CURVE_LABELS), ("recover", RECOVER_LABELS)):
        output = result[name]
        required = {"label", "index", "prob", "probs"}
        missing = required - set(output)
        if missing:
            raise AssertionError(f"{name} is missing fields: {sorted(missing)}")
        if output["label"] not in labels:
            raise AssertionError(f"invalid {name} label: {output['label']!r}")
        if not 0.0 <= output["prob"] <= 1.0:
            raise AssertionError(f"invalid {name} probability: {output['prob']}")
        probability_sum = sum(output["probs"].values())
        if abs(probability_sum - 1.0) > 1e-4:
            raise AssertionError(f"{name} probabilities sum to {probability_sum}, expected 1")


def predict(server_url: str, mode: str, sample: dict, timeout: float) -> tuple[dict, float]:
    if mode == "path":
        return request_json(
            f"{server_url}/predict",
            {"image": sample["image_path"]},
            timeout,
        )
    if mode == "base64":
        with open(sample["image_path"], "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return request_json(
            f"{server_url}/predict",
            {"image_bytes": encoded},
            timeout,
        )
    if mode == "file":
        return request_file(f"{server_url}/predict/file", sample["image_path"], timeout)
    raise ValueError(f"unsupported mode: {mode}")


def print_result(mode: str, result: dict, sample: dict, latency_ms: float) -> bool:
    trend_ok = result["trend"]["label"] == sample["curve"]
    recover_ok = result["recover"]["label"] == sample["recover"]
    print(f"\n[{mode}] latency={latency_ms:.2f} ms")
    print(
        f"  trend : prediction={result['trend']['label']:<10} "
        f"truth={sample['curve']:<10} confidence={result['trend']['prob']:.4f} "
        f"correct={trend_ok}"
    )
    print(
        f"  recover: prediction={result['recover']['label']:<10} "
        f"truth={sample['recover']:<10} confidence={result['recover']['prob']:.4f} "
        f"correct={recover_ok}"
    )
    return trend_ok and recover_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a running JEV FastAPI service")
    parser.add_argument("--server-url", default="http://127.0.0.1:8008")
    parser.add_argument("--mode", choices=("path", "base64", "file", "all"), default="base64")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--files-root", default=DEFAULT_FILES_ROOT)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--assert-labels",
        action="store_true",
        help="Exit with an assertion error when prediction differs from ground truth",
    )
    args = parser.parse_args()

    server_url = args.server_url.rstrip("/")
    health, health_ms = request(f"{server_url}/health", timeout=args.timeout)
    print(f"health ({health_ms:.2f} ms): {health}")
    if health.get("status") != "ok" or not health.get("model_loaded"):
        raise RuntimeError(f"JEV server is not ready: {health}")

    sample = load_sample(args.data_file, args.files_root, args.sample_index)
    print(f"sample={sample['index']} image={sample['image_path']}")

    modes = ("path", "base64", "file") if args.mode == "all" else (args.mode,)
    all_correct = True
    for mode in modes:
        result, latency_ms = predict(server_url, mode, sample, args.timeout)
        validate_prediction(result)
        all_correct &= print_result(mode, result, sample, latency_ms)

    if args.assert_labels and not all_correct:
        raise AssertionError("one or more predictions differ from ground truth")
    print("\nJEV server API test passed")


if __name__ == "__main__":
    main()
