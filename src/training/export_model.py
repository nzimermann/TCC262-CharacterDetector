"""Promote a training run's best.pt to the project's "official" model.

Copies <run>/weights/best.pt to models/character_detector.pt and
sanity-checks that the copy actually loads back with Ultralytics (task
detect, the expected 36 classes 0-9 + A-Z, in that exact id order) before
calling it done - a bad copy or a checkpoint from the wrong run is a bad
time to discover later, at inference time.

Usage:
    python src/training/export_model.py --weights outputs/runs/detect/full_run/weights/best.pt
    python src/training/export_model.py --weights outputs/runs/detect/smoke_test/weights/best.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "models" / "character_detector.pt"

EXPECTED_NAMES = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", required=True, help="ex: outputs/runs/detect/full_run/weights/best.pt"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    src = Path(args.weights)
    if not src.exists():
        raise RuntimeError(f"pesos não encontrados: {src}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, args.output)

    model = YOLO(str(args.output))
    size_mb = args.output.stat().st_size / (1024 * 1024)

    actual_names = [model.names[i] for i in sorted(model.names)]

    print(f"Copiado: {src} -> {args.output}")
    print(f"Tamanho: {size_mb:.1f} MB")
    print(f"Task: {model.task}")
    print(f"Classes ({len(actual_names)}): {actual_names}")

    if model.task != "detect" or actual_names != EXPECTED_NAMES:
        raise RuntimeError(
            "o modelo copiado não parece ser o detector de caracteres esperado "
            f"(task={model.task}, names={actual_names}) - confira se --weights aponta pro run certo."
        )

    print("\nOK - models/character_detector.pt pronto.")


if __name__ == "__main__":
    main()
