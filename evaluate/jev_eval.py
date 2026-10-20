"""Evaluate a trained JEV checkpoint on the Argos train or validation split.

Each Argos row is evaluated as one image inference containing two questions:
trend (7 candidates) and recover (4 candidates). Besides classification
metrics, the script reports preprocessing, batch preparation, model-forward,
and amortized end-to-end latency.
"""
from __future__ import annotations

import argparse
import json
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
    JEVDatasets,
    RECOVER_CHOICES,
    JevTorchDataset,
    argos_to_jev_records,
)
from infer.jev_infer import JevInference


DEFAULT_SPLITS = {
    "train": "datas/aiops/files/jsons/argos-trains.json",
    "valid": "datas/aiops/files/jsons/argos-valids.json",
}


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def classification_metrics(labels: list[str], truth: list[str], pred: list[str]) -> dict:
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for target, output in zip(truth, pred):
        matrix[target][output] += 1

    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[target][label] for target in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": sum(a == b for a, b in zip(truth, pred)) / len(truth),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "rows_are_ground_truth": [[matrix[a][b] for b in labels] for a in labels],
        },
    }


def confidence_metrics(confidences: list[float], correct: list[bool]) -> dict:
    good = [c for c, ok in zip(confidences, correct) if ok]
    bad = [c for c, ok in zip(confidences, correct) if not ok]
    return {
        "mean": sum(confidences) / len(confidences),
        "mean_correct": sum(good) / len(good) if good else None,
        "mean_incorrect": sum(bad) / len(bad) if bad else None,
    }


def build_dataset(rows: list[dict], engine: JevInference) -> JevTorchDataset:
    records = argos_to_jev_records(rows, engine.system)
    raw = JEVDatasets(
        data_files="",
        files_root=engine.files_root,
        check_images=True,
        answer_format="index",
    ).build(records)
    return JevTorchDataset(
        raw,
        engine.tokenizer,
        engine.image_processor,
        max_tokens=engine.max_tokens,
        max_pixels=engine.max_pixels,
    )


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=device.startswith("cuda"))
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def forward_batch(engine: JevInference, batch: dict) -> tuple[torch.Tensor, float, float]:
    device_type = "cuda" if engine.device.startswith("cuda") else "cpu"
    synchronize(engine.device)
    started = time.perf_counter()
    with torch.autocast(device_type=device_type, dtype=engine.amp_dtype):
        logits = engine.model(batch)["logits"].float()
    synchronize(engine.device)
    elapsed = time.perf_counter() - started

    mask = batch["cand_mask"]
    logits = logits.masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    probs = logp.exp() * mask
    target = batch["target"]
    ce_sum = float((-(target * logp).sum(-1)).sum())
    brier_sum = float((((probs - target) ** 2).sum(-1)).sum())
    return probs.cpu(), ce_sum, brier_sum, elapsed


def evaluate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    data_file = args.data_file or DEFAULT_SPLITS[args.split]
    with open(data_file, encoding="utf-8") as f:
        rows = json.load(f)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no records found in {data_file}")

    load_started = time.perf_counter()
    engine = JevInference(
        args.checkpoint,
        device=args.device,
        prompt_file=args.prompt_file or None,
        files_root=args.files_root,
    )
    model_load_seconds = time.perf_counter() - load_started

    warmup_dataset = build_dataset(rows[:1], engine)
    warmup_batch = move_batch(
        warmup_dataset.collate([warmup_dataset[0], warmup_dataset[1]]),
        engine.device,
    )
    for _ in range(args.warmup):
        forward_batch(engine, warmup_batch)
    del warmup_batch, warmup_dataset
    if engine.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(engine.device))

    truth = {"trend": [], "recover": []}
    pred = {"trend": [], "recover": []}
    confidences = {"trend": [], "recover": []}
    correct = {"trend": [], "recover": []}
    predictions = []
    prepare_seconds = 0.0
    forward_seconds = 0.0
    per_image_forward_ms = []
    ce_sum = 0.0
    brier_sum = 0.0
    preprocess_seconds = 0.0

    evaluation_started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))

        preprocess_started = time.perf_counter()
        batch_dataset = build_dataset(rows[start:end], engine)
        preprocess_seconds += time.perf_counter() - preprocess_started
        if len(batch_dataset) != 2 * (end - start):
            raise RuntimeError(
                "expected two questions per image, "
                f"got {len(batch_dataset)} questions for {end - start} images"
            )

        prepare_started = time.perf_counter()
        batch = move_batch(
            batch_dataset.collate([batch_dataset[i] for i in range(len(batch_dataset))]),
            engine.device,
        )
        synchronize(engine.device)
        prepare_seconds += time.perf_counter() - prepare_started

        probs, batch_ce, batch_brier, elapsed = forward_batch(engine, batch)
        forward_seconds += elapsed
        ce_sum += batch_ce
        brier_sum += batch_brier
        per_image_forward_ms.append(elapsed * 1000 / (end - start))

        for local_index, row in enumerate(rows[start:end]):
            trend_probs = probs[2 * local_index, : len(CURVE_CHOICES)]
            recover_probs = probs[2 * local_index + 1, : len(RECOVER_CHOICES)]
            trend_index = int(trend_probs.argmax())
            recover_index = int(recover_probs.argmax())
            trend_pred = CURVE_CHOICES[trend_index]
            recover_pred = RECOVER_CHOICES[recover_index]

            truth["trend"].append(row["curve"])
            truth["recover"].append(row["recover"])
            pred["trend"].append(trend_pred)
            pred["recover"].append(recover_pred)
            confidences["trend"].append(float(trend_probs[trend_index]))
            confidences["recover"].append(float(recover_probs[recover_index]))
            correct["trend"].append(trend_pred == row["curve"])
            correct["recover"].append(recover_pred == row["recover"])

            predictions.append({
                "index": start + local_index,
                "image": row["image"],
                "from": row["from"],
                "trend": {
                    "truth": row["curve"],
                    "prediction": trend_pred,
                    "confidence": float(trend_probs[trend_index]),
                    "probs": dict(zip(CURVE_CHOICES, map(float, trend_probs))),
                },
                "recover": {
                    "truth": row["recover"],
                    "prediction": recover_pred,
                    "confidence": float(recover_probs[recover_index]),
                    "probs": dict(zip(RECOVER_CHOICES, map(float, recover_probs))),
                },
            })
        del batch, batch_dataset

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
        "split": args.split,
        "data_file": data_file,
        "checkpoint": args.checkpoint,
        "checkpoint_step": engine.step,
        "device": engine.device,
        "images": image_count,
        "questions": question_count,
        "batch_size_images": args.batch_size,
        "loss": {
            "cross_entropy": ce_sum / question_count,
            "brier": brier_sum / question_count,
        },
        "trend": {
            **classification_metrics(CURVE_CHOICES, truth["trend"], pred["trend"]),
            "confidence": confidence_metrics(confidences["trend"], correct["trend"]),
        },
        "recover": {
            **classification_metrics(RECOVER_CHOICES, truth["recover"], pred["recover"]),
            "confidence": confidence_metrics(confidences["recover"], correct["recover"]),
        },
        "joint_accuracy": sum(joint_correct) / image_count,
        "efficiency": {
            "model_load_seconds": model_load_seconds,
            "dataset_preprocess_seconds": preprocess_seconds,
            "preprocess_ms_per_image": preprocess_seconds * 1000 / image_count,
            "batch_prepare_seconds": prepare_seconds,
            "forward_seconds": forward_seconds,
            "forward_ms_per_image": forward_seconds * 1000 / image_count,
            "forward_ms_per_image_p50": percentile(per_image_forward_ms, 0.50),
            "forward_ms_per_image_p95": percentile(per_image_forward_ms, 0.95),
            "evaluation_seconds": evaluation_seconds,
            "end_to_end_ms_per_image": evaluation_seconds * 1000 / image_count,
            "forward_images_per_second": image_count / forward_seconds,
            "end_to_end_images_per_second": image_count / evaluation_seconds,
            "cuda_peak_allocated_mb": peak_allocated_mb,
            "cuda_peak_reserved_mb": peak_reserved_mb,
            "warmup_iterations": args.warmup,
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
        f"latency: preprocess={speed['preprocess_ms_per_image']:.2f} ms/image, "
        f"forward={speed['forward_ms_per_image']:.2f} ms/image "
        f"(p50={speed['forward_ms_per_image_p50']:.2f}, "
        f"p95={speed['forward_ms_per_image_p95']:.2f}), "
        f"end_to_end={speed['end_to_end_ms_per_image']:.2f} ms/image"
    )
    print(
        f"throughput: forward={speed['forward_images_per_second']:.2f} images/s, "
        f"end_to_end={speed['end_to_end_images_per_second']:.2f} images/s"
    )
    if speed["cuda_peak_allocated_mb"] is not None:
        print(
            f"cuda_peak: allocated={speed['cuda_peak_allocated_mb']:.1f} MiB, "
            f"reserved={speed['cuda_peak_reserved_mb']:.1f} MiB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JEV on Argos train/validation data")
    parser.add_argument("--checkpoint", default="outputs/jev/best.pt")
    parser.add_argument("--split", choices=tuple(DEFAULT_SPLITS), default="valid")
    parser.add_argument("--data-file", default=None, help="Override the JSON selected by --split")
    parser.add_argument("--files-root", default=DEFAULT_FILES_ROOT)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--device", default=None, help="Default: CUDA when available, otherwise CPU")
    parser.add_argument("--batch-size", type=int, default=1, help="Images per forward; each image has two questions")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed model-forward warmup iterations")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    parser.add_argument("--output-file", default=None, help="Metrics JSON; defaults under outputs/jev_eval")
    parser.add_argument("--predictions-file", default=None, help="Optional detailed prediction JSON")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    metrics, predictions = evaluate(args)
    print_summary(metrics)

    output_file = args.output_file or os.path.join(
        "outputs", "jev_eval", f"{args.split}_metrics.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"metrics saved to {output_file}")

    if args.predictions_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.predictions_file)), exist_ok=True)
        with open(args.predictions_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"predictions saved to {args.predictions_file}")


if __name__ == "__main__":
    main()
