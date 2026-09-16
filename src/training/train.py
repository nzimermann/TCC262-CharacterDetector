"""Train the character detector (YOLO11n, 36 classes: 0-9 + A-Z).

Modes:
    python src/training/train.py                 # full run, reads configs/train.yaml
    python src/training/train.py --smoke-test     # tiny run, just to check the pipeline works
    python src/training/train.py --resume outputs/runs/detect/full_run/weights/last.pt

Uses every visible GPU via DDP automatically (e.g. Kaggle's "GPU T4 x2").
`configs/train.yaml`'s `batch` is the TOTAL batch, split across GPUs.
"""

import argparse
import shutil
import threading
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "yolo"
DATA_YAML = DATASET_DIR / "data.yaml"
TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"
RUNS_DIR = REPO_ROOT / "outputs" / "runs" / "detect"
SMOKE_DIR = REPO_ROOT / "data" / "yolo_smoke"
PRETRAINED_MODEL = REPO_ROOT / "weights" / "yolo11n.pt"


def build_smoke_subset(n_images: int) -> Path:
    """Hardlink a small (image, label) subset of train/ for a fast pipeline check."""
    src_images = DATASET_DIR / "images" / "train"
    src_labels = DATASET_DIR / "labels" / "train"

    label_files = sorted(src_labels.glob("*.txt"))[:n_images]
    if len(label_files) < n_images:
        raise RuntimeError(
            f"só achei {len(label_files)} labels em {src_labels} - rode build_dataset.py primeiro."
        )

    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    for split in ("train", "val"):
        (SMOKE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (SMOKE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for label_path in label_files:
        image_path = src_images / f"{label_path.stem}.jpg"
        for split in ("train", "val"):
            try:
                (SMOKE_DIR / "labels" / split / label_path.name).hardlink_to(label_path)
                (SMOKE_DIR / "images" / split / image_path.name).hardlink_to(image_path)
            except OSError:
                shutil.copy2(label_path, SMOKE_DIR / "labels" / split / label_path.name)
                shutil.copy2(image_path, SMOKE_DIR / "images" / split / image_path.name)

    with open(DATA_YAML, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    (SMOKE_DIR / "data.yaml").write_text(
        f"path: {SMOKE_DIR.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {full_cfg['nc']}\nnames: {full_cfg['names']}\n",
        encoding="utf-8",
    )
    return SMOKE_DIR / "data.yaml"


def watch_checkpoints(
    weights_dir: Path, live_copy_dir: Path | None, stop_event: threading.Event
):
    """Print + copy last.pt/best.pt whenever last.pt changes on disk.

    Polls the file instead of using Ultralytics' on_model_save callback:
    that callback never fires on multi-GPU runs, since DDP trains in a
    separate subprocess that doesn't know about it.
    """
    start = time.time()
    last_mtime = None
    while not stop_event.is_set():
        last_pt = weights_dir / "last.pt"
        mtime = last_pt.stat().st_mtime if last_pt.exists() else None
        if mtime and mtime != last_mtime:
            last_mtime = mtime
            print(
                f"[checkpoint] last.pt atualizado ({(time.time() - start) / 60:.1f} min) -> {last_pt}",
                flush=True,
            )
            if live_copy_dir:
                live_copy_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(last_pt, live_copy_dir / "last.pt")
                best_pt = weights_dir / "best.pt"
                if best_pt.exists():
                    shutil.copy2(best_pt, live_copy_dir / "best.pt")
        stop_event.wait(5.0)


def train_with_checkpoint_watch(
    model: YOLO, train_kwargs: dict, weights_dir: Path, live_copy_dir: Path | None
):
    """model.train() with watch_checkpoints polling in the background."""
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=watch_checkpoints,
        args=(weights_dir, live_copy_dir, stop_event),
        daemon=True,
    )
    watcher.start()
    try:
        model.train(**train_kwargs)
    finally:
        stop_event.set()
        watcher.join(timeout=10)


def report_result(model: YOLO):
    save_dir = getattr(model.trainer, "save_dir", None)
    best_pt = Path(save_dir) / "weights" / "best.pt" if save_dir else None
    if best_pt and best_pt.exists():
        print(f"\nOK - modelo salvo em: {best_pt}")
    else:
        print(f"\n[aviso] não encontrei best.pt - confira {save_dir}")


def pick_device() -> tuple[int | str | list[int], int]:
    if not torch.cuda.is_available():
        return "cpu", 0
    n_gpus = torch.cuda.device_count()
    return (list(range(n_gpus)) if n_gpus > 1 else 0), n_gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n", type=int, default=10, help="imagens no smoke test")
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="batch do smoke test (default: min(4, --n))",
    )
    parser.add_argument(
        "--live-copy-dir",
        type=Path,
        default=None,
        help="ex: /kaggle/working/checkpoints",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="last.pt de um run interrompido"
    )
    args = parser.parse_args()

    device, n_gpus = pick_device()
    print(f"Dispositivo: {device} (GPUs: {n_gpus})")

    if args.resume:
        if not args.resume.exists():
            raise RuntimeError(f"checkpoint não encontrado: {args.resume}")
        # um checkpoint de treino já finalizado tem epoch=-1; resume=True nesse
        # caso não avisa, só treina um run novo do zero silenciosamente
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("epoch", -1) < 0:
            raise RuntimeError(
                f"{args.resume} já é de um treino completo - nada a retomar."
            )

        model = YOLO(str(args.resume))
        train_with_checkpoint_watch(
            model, {"resume": str(args.resume)}, args.resume.parent, args.live_copy_dir
        )
        report_result(model)
        return

    if args.smoke_test:
        print(
            f"*** SMOKE TEST: {args.n} imagens, {args.smoke_epochs} época(s) - métricas não significam nada ***"
        )
        run_name = "smoke_test"
        model_name = str(PRETRAINED_MODEL)
        train_kwargs = dict(
            data=str(build_smoke_subset(args.n)),
            epochs=args.smoke_epochs,
            imgsz=640,
            batch=args.batch or min(4, args.n),
            device=device,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            patience=0,
            plots=False,
        )
    else:
        with open(TRAIN_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run_name = cfg["name"]
        model_name = str(REPO_ROOT / cfg["model"])
        train_kwargs = dict(
            data=str(DATA_YAML),
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=device,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            patience=cfg["patience"],
            save_period=cfg.get("save_period", -1),
        )

    model = YOLO(model_name)
    train_with_checkpoint_watch(
        model, train_kwargs, RUNS_DIR / run_name / "weights", args.live_copy_dir
    )
    report_result(model)


if __name__ == "__main__":
    main()
