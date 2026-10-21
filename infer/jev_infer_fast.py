"""Command-line interface for the optimized JEV inference runtime."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infer.jev_runtime import FastJevInference

DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "outputs/jev/best.pt")
DEFAULT_PROMPT_FILE = os.path.join(
    REPO_ROOT,
    "datas/aiops/files/prompts/argos-dp-prompt.md",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized JEV single-image inference")
    parser.add_argument("image")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("JEV_CHECKPOINT", DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--backbone", default=os.environ.get("JEV_BACKBONE"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")

    engine = FastJevInference(
        args.checkpoint,
        backbone=args.backbone,
        device=args.device,
        prompt_file=args.prompt_file or None,
    )
    for _ in range(args.warmup):
        engine.predict_image(args.image)

    latencies = []
    result = None
    for _ in range(args.repeat):
        result = engine.predict_image(args.image)
        latencies.append(engine.last_latency_ms)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for task in ("trend", "recover"):
            answer = result[task]
            print(f"[{task}] {answer['label']}  (prob={answer['prob']:.3f})")
            for label, probability in sorted(
                answer["probs"].items(),
                key=lambda item: -item[1],
            ):
                print(f"    {label:<12} {probability:.4f}")
    print(
        f"latency_ms: median={statistics.median(latencies):.2f} "
        f"min={min(latencies):.2f} max={max(latencies):.2f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
