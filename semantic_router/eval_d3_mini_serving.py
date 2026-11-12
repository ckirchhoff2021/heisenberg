#!/usr/bin/env python3
"""Evaluate and benchmark a d3-mini SystemOne HTTP service.

The script evaluates a deployed SystemOne server on the local ``jev-mixed``
dataset and measures request throughput / end-to-end latency at several
concurrency levels.  It uses only the Python standard library.

Default endpoint:
    http://127.0.0.1:8899/v1/systemone

Example:
    python semantic_router/eval_d3_mini_serving.py \
        --eval-concurrency 8 \
        --benchmark-limit 256 \
        --concurrency-list 1 4 8 16
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIXED = REPO_ROOT / "datas/aiops/noise/jev-mixed.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/semantic_router_serving"
DEFAULT_ENDPOINT = "http://127.0.0.1:8899/v1/systemone"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy and performance of a served d3-mini model."
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="SystemOne endpoint URL")
    parser.add_argument("--model", default="d3-mini", help="served model ID")
    parser.add_argument("--mixed-file", default=str(DEFAULT_MIXED),
                        help="jev-mixed dataset JSON path")
    parser.add_argument("--split", default="eval", help="dataset split")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N records before sampling")
    parser.add_argument("--eval-concurrency", type=int, default=8,
                        help="parallel clients for the full accuracy evaluation")
    parser.add_argument("--benchmark-limit", type=int, default=256,
                        help="representative records used by each throughput test")
    parser.add_argument("--concurrency-list", type=int, nargs="+",
                        default=[1, 4, 8, 16],
                        help="client concurrency levels for throughput tests")
    parser.add_argument("--warmup", type=int, default=8,
                        help="untimed warmup requests before benchmarking")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="per-request timeout in seconds")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="directory for metrics and reports")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="only evaluate accuracy")
    return parser.parse_args()


def load_samples(dataset_path: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    schema = data["schema"]
    records = data["splits"][split]
    if limit is not None:
        records = records[:limit]

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        task_id = record["task_id"]
        task = schema["tasks"][task_id]
        choices = task["choices"]
        image_rel = record["state"]["images"][0]
        data_root = schema["data_roots"][task["data_root_ref"]]
        image_path = Path(data_root) / image_rel
        samples.append(
            {
                "index": index,
                "task_id": task_id,
                "state": schema["systems"][task["system_ref"]],
                "question": task["question"],
                "choices": choices,
                "answer_index": int(record["answer"]),
                "answer": choices[int(record["answer"])],
                "image_path": str(image_path),
            }
        )
    return samples


def image_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    raw = Path(path).read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(sample: dict[str, Any], model: str) -> dict[str, Any]:
    choices = sample["choices"]
    return {
        "model": model,
        "state": sample["state"],
        "images": [image_data_url(sample["image_path"])],
        "questions": {
            "q": {
                "type": "choice",
                "instructions": sample["question"],
                "criteria": {label: None for label in choices},
            }
        },
    }


_thread_local = threading.local()


def opener() -> urllib.request.OpenerDirector:
    if not hasattr(_thread_local, "opener"):
        # Do not install globally; each client thread retains its own keep-alive handler.
        _thread_local.opener = urllib.request.build_opener()
    return _thread_local.opener


def post_json(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener().open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from None


def run_one(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        response = post_json(
            args.endpoint, build_payload(sample, args.model), args.timeout
        )
        answer = response["answers"]["q"]
        prediction = answer["choice"]
        usage = response.get("usage", {})
        latency = time.perf_counter() - start
        return {
            "index": sample["index"],
            "task_id": sample["task_id"],
            "label": sample["answer"],
            "prediction": prediction,
            "correct": prediction == sample["answer"],
            "confidence": answer.get("confidence"),
            "probabilities": answer.get("probabilities"),
            "latency_s": latency,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - record failed requests in metrics
        return {
            "index": sample["index"],
            "task_id": sample["task_id"],
            "label": sample["answer"],
            "prediction": None,
            "correct": False,
            "confidence": None,
            "probabilities": None,
            "latency_s": time.perf_counter() - start,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(exc),
        }


def run_requests(
    samples: list[dict[str, Any]],
    concurrency: int,
    args: argparse.Namespace,
    desc: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, sample, args) for sample in samples]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - wall_start
    results.sort(key=lambda item: item["index"])

    successes = [item for item in results if item["error"] is None]
    latencies = [float(item["latency_s"]) for item in successes]
    input_tokens = sum(int(item["input_tokens"]) for item in successes)
    return {
        "desc": desc,
        "concurrency": concurrency,
        "total": len(results),
        "succeeded": len(successes),
        "failed": len(results) - len(successes),
        "wall_s": wall,
        "requests_per_s": len(successes) / wall if wall else 0.0,
        "input_tokens": input_tokens,
        "input_tokens_per_s": input_tokens / wall if wall else 0.0,
        "latency_s": summarize_latency(latencies),
        "results": results,
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
                "max": 0.0, "stdev": 0.0}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "stdev": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def empty_class_stats() -> dict[str, int]:
    return {"gold": 0, "pred": 0, "tp": 0}


def calculate_accuracy(results: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    choices_by_task = {sample["task_id"]: sample["choices"] for sample in samples}
    task_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "failed": 0,
                 "classes": defaultdict(empty_class_stats)}
    )

    for item in results:
        task_id = item["task_id"]
        metric = task_metrics[task_id]
        metric["total"] += 1
        if item["error"] is not None:
            metric["failed"] += 1
        if item["correct"]:
            metric["correct"] += 1

        label = item["label"]
        prediction = item["prediction"]
        metric["classes"][label]["gold"] += 1
        if prediction is not None:
            metric["classes"][prediction]["pred"] += 1
        if item["correct"]:
            metric["classes"][label]["tp"] += 1

    per_task: dict[str, Any] = {}
    total = correct = failed = 0
    for task_id in sorted(task_metrics):
        metric = task_metrics[task_id]
        per_class = {}
        weighted_f1_num = 0.0
        f1_values = []

        # Ensure every legal choice is represented, even if absent in this split.
        labels = set(choices_by_task[task_id]) | set(metric["classes"])
        for label in sorted(labels):
            stats = metric["classes"][label]
            precision = stats["tp"] / stats["pred"] if stats["pred"] else 0.0
            recall = stats["tp"] / stats["gold"] if stats["gold"] else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0 else 0.0
            )
            per_class[label] = {
                "support": stats["gold"],
                "predicted": stats["pred"],
                "correct": stats["tp"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            if stats["gold"]:
                weighted_f1_num += f1 * stats["gold"]
            if stats["gold"] or stats["pred"]:
                f1_values.append(f1)

        task_total = metric["total"]
        task_acc = metric["correct"] / task_total if task_total else 0.0
        per_task[task_id] = {
            "total": task_total,
            "correct": metric["correct"],
            "failed": metric["failed"],
            "acc": task_acc,
            "weighted_f1": weighted_f1_num / task_total if task_total else 0.0,
            "macro_f1": statistics.fmean(f1_values) if f1_values else 0.0,
            "per_class": per_class,
        }
        total += task_total
        correct += metric["correct"]
        failed += metric["failed"]

    return {
        "total": total,
        "correct": correct,
        "failed": failed,
        "acc": correct / total if total else 0.0,
        "per_task": per_task,
    }


def representative_subset(samples: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(samples):
        return samples
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["task_id"]].append(sample)
    selected: list[dict[str, Any]] = []
    task_ids = sorted(groups)
    while len(selected) < size:
        progressed = False
        for task_id in task_ids:
            group = groups[task_id]
            offset = sum(1 for item in selected if item["task_id"] == task_id)
            if offset < len(group):
                selected.append(group[offset])
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            break
    return selected


def check_service(args: argparse.Namespace) -> None:
    base = args.endpoint.rsplit("/", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=10) as response:
            models = json.loads(response.read().decode("utf-8"))
        available = {item.get("id") for item in models.get("data", [])}
        if args.model not in available:
            print(f"Warning: model '{args.model}' not reported in {sorted(available)}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: service check failed: {exc}")


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    accuracy_run: dict[str, Any],
    accuracy: dict[str, Any],
    benchmark_runs: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "endpoint": args.endpoint,
        "model": args.model,
        "dataset": os.path.abspath(args.mixed_file),
        "split": args.split,
        "accuracy_wall_s": accuracy_run["wall_s"],
        "accuracy": {key: value for key, value in accuracy.items()},
        "benchmarks": [
            {key: value for key, value in run.items() if key != "results"}
            for run in benchmark_runs
        ],
        "accuracy_requests": accuracy_run["results"],
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    lines = [
        "# d3-mini 服务评测报告",
        "",
        f"- Endpoint: `{args.endpoint}`",
        f"- Model: `{args.model}`",
        f"- Dataset: `{args.mixed_file}` ({args.split})",
        f"- Samples: {accuracy['total']}",
        f"- Accuracy wall time: {accuracy_run['wall_s']:.2f}s",
        "",
        "## 1. Accuracy",
        "",
        f"Overall accuracy: **{accuracy['acc']:.4f}** "
        f"({accuracy['correct']}/{accuracy['total']}), failed requests: {accuracy['failed']}",
        "",
        "| Task | Acc | Correct | Total | Failed | Weighted F1 | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task_id, task in accuracy["per_task"].items():
        lines.append(
            f"| {task_id} | {task['acc']:.4f} | {task['correct']} | "
            f"{task['total']} | {task['failed']} | {task['weighted_f1']:.4f} | "
            f"{task['macro_f1']:.4f} |"
        )

    lines.extend(["", "## 2. Throughput and latency", "",
                  "| Concurrency | OK | Fail | Wall(s) | Req/s | Input tok/s | "
                  "Mean(s) | P50(s) | P90(s) | P95(s) | P99(s) | Max(s) |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for run in benchmark_runs:
        latency = run["latency_s"]
        lines.append(
            f"| {run['concurrency']} | {run['succeeded']} | {run['failed']} | "
            f"{run['wall_s']:.2f} | {run['requests_per_s']:.2f} | "
            f"{run['input_tokens_per_s']:.1f} | {latency['mean']:.3f} | "
            f"{latency['p50']:.3f} | {latency['p90']:.3f} | "
            f"{latency['p95']:.3f} | {latency['p99']:.3f} | {latency['max']:.3f} |"
        )

    lines.extend([
        "",
        "Notes:",
        "- Latency is client-observed HTTP request wall time, including image base64 transfer,",
        "  server preprocessing/model execution, and queueing under concurrency.",
        "- `output_tokens` is zero for this decision head; requests/s is the primary throughput metric.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.mixed_file)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    samples = load_samples(dataset_path, args.split, args.limit)
    if not samples:
        print("No samples to evaluate.", file=sys.stderr)
        return 1

    print(f"Loaded {len(samples)} samples from {dataset_path}")
    check_service(args)

    print(f"Evaluating accuracy with {args.eval_concurrency} clients ...")
    accuracy_run = run_requests(
        samples, args.eval_concurrency, args, "full-accuracy-evaluation"
    )
    accuracy = calculate_accuracy(accuracy_run["results"], samples)
    print(f"Accuracy: {accuracy['acc']:.4f} "
          f"({accuracy['correct']}/{accuracy['total']})")

    benchmark_runs: list[dict[str, Any]] = []
    if not args.skip_benchmark:
        benchmark_samples = representative_subset(samples, args.benchmark_limit)
        if args.warmup:
            warmup_samples = benchmark_samples[: min(args.warmup, len(benchmark_samples))]
            print(f"Running {len(warmup_samples)} warmup requests ...")
            run_requests(warmup_samples, min(4, len(warmup_samples)), args, "warmup")

        for concurrency in args.concurrency_list:
            print(f"Benchmarking concurrency={concurrency} "
                  f"on {len(benchmark_samples)} samples ...")
            run = run_requests(
                benchmark_samples, concurrency, args,
                f"benchmark-c{concurrency}"
            )
            benchmark_runs.append(run)
            print(f"  {run['requests_per_s']:.2f} req/s, "
                  f"p50={run['latency_s']['p50']:.3f}s, "
                  f"p99={run['latency_s']['p99']:.3f}s")

    output_dir = Path(args.output_dir)
    write_report(output_dir, args, accuracy_run, accuracy, benchmark_runs)
    print(f"Report written to {output_dir / 'REPORT.md'}")
    print(f"Detailed metrics written to {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
