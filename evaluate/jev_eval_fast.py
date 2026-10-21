"""Evaluate the optimized JEV inference path on Argos train/validation data.

Unlike ``evaluate/jev_eval.py``, this script calls
``FastJevInference.predict_image`` for each image. The reported latency is
therefore the optimized single-image path, including image preprocessing,
vision encoding, cached language-model prefixes, candidate scoring, and result
conversion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.jev import (
    CURVE_CHOICES,
    DEFAULT_FILES_ROOT,
    DEFAULT_PROMPT_FILE,
    RECOVER_CHOICES,
)
from evaluate.jev_eval import (
    DEFAULT_SPLITS,
    classification_metrics,
    confidence_metrics,
    percentile,
    synchronize,
)
from infer.jev_runtime import FastJevInference


def resolve_image_path(row: dict, files_root: str) -> str:
    image = row.get("image")
    if not image:
        raise ValueError("missing image")

    candidates = [image] if os.path.isabs(image) else [
        os.path.join(files_root, row.get("from", ""), image),
        os.path.join(files_root, image),
        image,
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"image not found; tried: {candidates}")


def ordered_probabilities(answer: dict, labels: list[str]) -> list[float]:
    probabilities = answer.get("probs", {})
    missing = [label for label in labels if label not in probabilities]
    if missing:
        raise ValueError(f"prediction is missing probabilities for: {missing}")
    values = [float(probabilities[label]) for label in labels]
    total = sum(values)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-3:
        raise ValueError(f"invalid probability sum: {total}")
    return values


def question_loss(probabilities: list[float], labels: list[str], truth: str) -> tuple[float, float]:
    try:
        target = labels.index(truth)
    except ValueError as error:
        raise ValueError(f"unknown ground-truth label {truth!r}; expected one of {labels}") from error
    ce = -math.log(max(probabilities[target], 1e-12))
    brier = sum(
        (probability - float(index == target)) ** 2
        for index, probability in enumerate(probabilities)
    )
    return ce, brier


def prediction_margin(probabilities: list[float]) -> float:
    top = sorted(probabilities, reverse=True)
    return top[0] - top[1]


def margin_metrics(margins: list[float]) -> dict:
    return {
        "mean": sum(margins) / len(margins),
        "p05": percentile(margins, 0.05),
        "p50": percentile(margins, 0.50),
        "below_0_01": sum(margin < 0.01 for margin in margins),
        "below_0_05": sum(margin < 0.05 for margin in margins),
    }


def evaluate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    data_file = args.data_file or DEFAULT_SPLITS[args.split]
    with open(data_file, encoding="utf-8") as file:
        rows = json.load(file)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no records found in {data_file}")

    model_load_started = time.perf_counter()
    engine = FastJevInference(
        args.checkpoint,
        backbone=args.backbone,
        device=args.device,
        prompt_file=args.prompt_file or None,
    )
    model_load_seconds = time.perf_counter() - model_load_started

    first_image = resolve_image_path(rows[0], args.files_root)
    for _ in range(args.warmup):
        engine.predict_image(first_image)
    if engine.device.startswith("cuda"):
        synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(torch.device(engine.device))

    truth = {"trend": [], "recover": []}
    pred = {"trend": [], "recover": []}
    confidences = {"trend": [], "recover": []}
    correct = {"trend": [], "recover": []}
    margins = {"trend": [], "recover": []}
    latency_ms = []
    predictions = []
    ce_sum = 0.0
    brier_sum = 0.0

    evaluation_started = time.perf_counter()
    for index, row in enumerate(rows):
        image_path = resolve_image_path(row, args.files_root)
        result = engine.predict_image(image_path)
        elapsed_ms = float(engine.last_latency_ms)
        latency_ms.append(elapsed_ms)

        row_prediction = {
            "index": index,
            "image": row["image"],
            "from": row.get("from"),
            "latency_ms": elapsed_ms,
        }
        for task, labels, target in (
            ("trend", CURVE_CHOICES, row.get("curve")),
            ("recover", RECOVER_CHOICES, row.get("recover")),
        ):
            answer = result[task]
            probabilities = ordered_probabilities(answer, labels)
            task_ce, task_brier = question_loss(probabilities, labels, target)
            ce_sum += task_ce
            brier_sum += task_brier

            output = str(answer["label"])
            confidence = float(answer["prob"])
            margin = prediction_margin(probabilities)
            truth[task].append(target)
            pred[task].append(output)
            confidences[task].append(confidence)
            correct[task].append(output == target)
            margins[task].append(margin)
            row_prediction[task] = {
                "truth": target,
                "prediction": output,
                "confidence": confidence,
                "margin": margin,
                "probs": dict(zip(labels, probabilities)),
            }
        predictions.append(row_prediction)

        completed = index + 1
        if args.log_every and (completed % args.log_every == 0 or completed == len(rows)):
            elapsed = time.perf_counter() - evaluation_started
            print(
                f"[{completed}/{len(rows)}] "
                f"avg={elapsed * 1000 / completed:.2f} ms/image "
                f"trend_acc={sum(correct['trend']) / completed:.4f} "
                f"recover_acc={sum(correct['recover']) / completed:.4f}",
                flush=True,
            )

    synchronize(engine.device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    image_count = len(rows)
    question_count = 2 * image_count
    joint_correct = [
        trend_ok and recover_ok
        for trend_ok, recover_ok in zip(correct["trend"], correct["recover"])
    ]

    peak_allocated_mb = None
    peak_reserved_mb = None
    if engine.device.startswith("cuda"):
        cuda_device = torch.device(engine.device)
        peak_allocated_mb = torch.cuda.max_memory_allocated(cuda_device) / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved(cuda_device) / 1024**2

    metrics = {
        "engine": "fast_prefix_cache",
        "split": args.split,
        "data_file": data_file,
        "checkpoint": args.checkpoint,
        "checkpoint_step": engine.step,
        "device": engine.device,
        "images": image_count,
        "questions": question_count,
        "loss": {
            "cross_entropy": ce_sum / question_count,
            "brier": brier_sum / question_count,
        },
        "trend": {
            **classification_metrics(CURVE_CHOICES, truth["trend"], pred["trend"]),
            "confidence": confidence_metrics(confidences["trend"], correct["trend"]),
            "margin": margin_metrics(margins["trend"]),
        },
        "recover": {
            **classification_metrics(RECOVER_CHOICES, truth["recover"], pred["recover"]),
            "confidence": confidence_metrics(confidences["recover"], correct["recover"]),
            "margin": margin_metrics(margins["recover"]),
        },
        "joint_accuracy": sum(joint_correct) / image_count,
        "efficiency": {
            "model_load_seconds": model_load_seconds,
            "warmup_iterations": args.warmup,
            "inference_seconds": sum(latency_ms) / 1000,
            "inference_ms_per_image": sum(latency_ms) / image_count,
            "inference_ms_per_image_min": min(latency_ms),
            "inference_ms_per_image_p50": percentile(latency_ms, 0.50),
            "inference_ms_per_image_p95": percentile(latency_ms, 0.95),
            "inference_ms_per_image_max": max(latency_ms),
            "evaluation_seconds": evaluation_seconds,
            "end_to_end_ms_per_image": evaluation_seconds * 1000 / image_count,
            "inference_images_per_second": image_count / (sum(latency_ms) / 1000),
            "end_to_end_images_per_second": image_count / evaluation_seconds,
            "cuda_peak_allocated_mb": peak_allocated_mb,
            "cuda_peak_reserved_mb": peak_reserved_mb,
        },
    }
    return metrics, predictions


def print_summary(metrics: dict) -> None:
    speed = metrics["efficiency"]
    print(
        f"images={metrics['images']} questions={metrics['questions']} "
        f"CE={metrics['loss']['cross_entropy']:.4f} "
        f"Brier={metrics['loss']['brier']:.4f}"
    )
    print(
        f"trend_acc={metrics['trend']['accuracy']:.4f} "
        f"recover_acc={metrics['recover']['accuracy']:.4f} "
        f"joint_acc={metrics['joint_accuracy']:.4f}"
    )
    print(
        f"latency: mean={speed['inference_ms_per_image']:.2f} ms/image "
        f"(p50={speed['inference_ms_per_image_p50']:.2f}, "
        f"p95={speed['inference_ms_per_image_p95']:.2f}), "
        f"end_to_end={speed['end_to_end_ms_per_image']:.2f} ms/image"
    )
    print(
        f"throughput: inference={speed['inference_images_per_second']:.2f} images/s, "
        f"end_to_end={speed['end_to_end_images_per_second']:.2f} images/s"
    )
    if speed["cuda_peak_allocated_mb"] is not None:
        print(
            f"cuda_peak: allocated={speed['cuda_peak_allocated_mb']:.1f} MiB, "
            f"reserved={speed['cuda_peak_reserved_mb']:.1f} MiB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate optimized JEV inference on Argos train/validation data"
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("JEV_CHECKPOINT", "outputs/jev/best.pt"),
    )
    parser.add_argument("--backbone", default=os.environ.get("JEV_BACKBONE"))
    parser.add_argument("--split", choices=tuple(DEFAULT_SPLITS), default="valid")
    parser.add_argument("--data-file", default=None, help="Override the JSON selected by --split")
    parser.add_argument("--files-root", default=DEFAULT_FILES_ROOT)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--device", default=None, help="Default: CUDA when available, otherwise CPU")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed single-image warmup iterations")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    parser.add_argument("--log-every", type=int, default=20, help="Progress interval; 0 disables")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Metrics JSON; defaults under outputs/jev_eval_fast",
    )
    parser.add_argument("--predictions-file", default=None, help="Optional detailed prediction JSON")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.log_every < 0:
        parser.error("--log-every must be >= 0")

    metrics, predictions = evaluate(args)
    print_summary(metrics)

    output_file = args.output_file or os.path.join(
        "outputs",
        "jev_eval_fast",
        f"{args.split}_metrics.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f"metrics saved to {output_file}")

    if args.predictions_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.predictions_file)), exist_ok=True)
        with open(args.predictions_file, "w", encoding="utf-8") as file:
            json.dump(predictions, file, ensure_ascii=False, indent=2)
        print(f"predictions saved to {args.predictions_file}")


if __name__ == "__main__":
    main()
