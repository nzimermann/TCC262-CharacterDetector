"""Train the character detector (YOLO11n, 36 classes: 0-9 + A-Z).

Two modes, controlled by ONE toggle (edit the constant below, or pass
--smoke-test on the command line - either works):

  - SMOKE_TEST = False (default): real run. Reads hyperparameters from
    configs/train.yaml (imgsz=640, epochs=..., batch=32, ...) and trains
    on the full data/yolo/ split (16,565 train images). Meant to run on
    Kaggle/Colab - this machine has no NVIDIA GPU.

  - SMOKE_TEST = True: fast pipeline check. Builds a tiny subset
    (SMOKE_TEST_N_IMAGES images, default 10) from the already-built
    data/yolo/images/train + labels/train, reusing the REAL 36 classes
    from data/yolo/data.yaml (not a dummy 1-class file), and trains for
    SMOKE_TEST_EPOCHS epochs (default 3). The only goal is confirming the
    pipeline runs start to finish and produces real .pt checkpoints along
    the way - the resulting model is not meant to be good. Train and val
    use the SAME tiny subset on purpose (there's no intention to measure
    real generalization here), so any metric printed in this mode is
    meaningless and should be ignored. On a Kaggle GPU, bump --n and
    --smoke-epochs to get a realistic per-epoch timing reading (and check
    VRAM headroom for configs/train.yaml's batch=32) before committing to
    a long run.

Safety nets for long unattended Kaggle runs (same lesson learned the hard
way on the PlateDetector project - a session hit the ~12h cap with
nothing recoverable):
  - Every run prints a line after each epoch confirming last.pt/best.pt
    were just written, with elapsed time.
  - --live-copy-dir: also copies last.pt/best.pt to a flat directory of
    your choice after every epoch, in addition to Ultralytics' own
    outputs/runs/detect/<name>/weights/.
  - --resume: continue an interrupted run from its last.pt, instead of
    starting over. Needs the checkpoint's *whole* run folder (last.pt +
    the args.yaml next to it), not just the .pt file alone.

Usage:
    python src/training/train.py                 # full run (needs configs/train.yaml)
    python src/training/train.py --smoke-test     # fast local sanity check (CPU is fine)
    python src/training/train.py --smoke-test --n 300 --smoke-epochs 5 --live-copy-dir /kaggle/working/checkpoints
    python src/training/train.py --resume outputs/runs/detect/full_run/weights/last.pt
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# --- the one toggle: flip this OR pass --smoke-test, both do the same thing ---
SMOKE_TEST = False
SMOKE_TEST_N_IMAGES = 10
SMOKE_TEST_EPOCHS = 3

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "yolo"
DATA_YAML = DATASET_DIR / "data.yaml"
TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"
RUNS_DIR = REPO_ROOT / "outputs" / "runs" / "detect"

SMOKE_DIR = REPO_ROOT / "data" / "yolo_smoke"
PRETRAINED_MODEL = REPO_ROOT / "weights" / "yolo11n.pt"


def build_smoke_subset(n_images: int) -> Path:
    """Hardlink `n_images` (image, label) pairs from the real train split
    into a tiny throwaway dataset, used for both train and val. Reuses the
    real nc/names from data/yolo/data.yaml so the smoke run exercises the
    actual 36-class head, not a placeholder. Returns the path to the small
    data.yaml describing it."""
    src_images = DATASET_DIR / "images" / "train"
    src_labels = DATASET_DIR / "labels" / "train"

    label_files = sorted(src_labels.glob("*.txt"))[:n_images]
    if len(label_files) < n_images:
        raise RuntimeError(
            f"pedi {n_images} imagens para o smoke test, mas só achei {len(label_files)} "
            f"em {src_labels} - rode build_dataset.py primeiro."
        )

    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)

    for split in ("train", "val"):
        (SMOKE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (SMOKE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for label_path in label_files:
        stem = label_path.stem
        image_path = src_images / f"{stem}.jpg"
        for split in ("train", "val"):
            try:
                (SMOKE_DIR / "labels" / split / label_path.name).hardlink_to(label_path)
                (SMOKE_DIR / "images" / split / image_path.name).hardlink_to(image_path)
            except OSError:
                shutil.copy2(label_path, SMOKE_DIR / "labels" / split / label_path.name)
                shutil.copy2(image_path, SMOKE_DIR / "images" / split / image_path.name)

    if not DATA_YAML.exists():
        raise RuntimeError(f"{DATA_YAML} não existe - rode build_dataset.py primeiro.")
    with open(DATA_YAML, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    yaml_path = SMOKE_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {SMOKE_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {full_cfg['nc']}\n"
        f"names: {full_cfg['names']}\n",
        encoding="utf-8",
    )
    return yaml_path


def make_checkpoint_callback(live_copy_dir: Path | None):
    """Prints proof-of-life after every epoch and, if `live_copy_dir` is
    set, also copies last.pt/best.pt there - so checkpoints are easy to
    find and grab manually mid-run without digging through outputs/runs/detect/."""
    start_time = time.time()

    def on_model_save(trainer):
        epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed while training
        elapsed_min = (time.time() - start_time) / 60
        weights_dir = Path(trainer.save_dir) / "weights"
        last_pt = weights_dir / "last.pt"

        msg = f"[checkpoint] epoca {epoch} concluida ({elapsed_min:.1f} min desde o inicio) -> {last_pt}"

        if live_copy_dir is not None and last_pt.exists():
            live_copy_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(last_pt, live_copy_dir / "last.pt")
            best_pt = weights_dir / "best.pt"
            if best_pt.exists():
                shutil.copy2(best_pt, live_copy_dir / "best.pt")
            msg += f" (copiado tambem para {live_copy_dir})"

        print(msg, flush=True)

    return on_model_save


def report_result(model):
    trainer = model.trainer
    save_dir = getattr(trainer, "save_dir", None)
    if not isinstance(save_dir, (str, Path)):
        print("\n[aviso] treino terminou mas o diretório de saída não foi informado")
        return

    best_pt = Path(save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\nOK - modelo salvo em: {best_pt}")
    else:
        print(
            f"\n[aviso] treino terminou mas não encontrei {best_pt} - confira {save_dir}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", default=SMOKE_TEST)
    parser.add_argument(
        "--n", type=int, default=SMOKE_TEST_N_IMAGES, help="imagens no smoke test"
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=SMOKE_TEST_EPOCHS,
        help="épocas no smoke test",
    )
    parser.add_argument(
        "--live-copy-dir",
        type=Path,
        default=None,
        help="também copia last.pt/best.pt pra essa pasta a cada época (ex: /kaggle/working/checkpoints)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continua um treino interrompido a partir do last.pt dele (precisa da pasta do run inteira, "
        "com o args.yaml do lado - não só o .pt sozinho). Ignora --smoke-test e configs/train.yaml.",
    )
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} (cuda disponível: {torch.cuda.is_available()})")

    if args.resume:
        if not args.resume.exists():
            raise RuntimeError(f"checkpoint de resume não encontrado: {args.resume}")

        # Ultralytics marks a checkpoint's saved epoch as -1 once that run
        # finished all its epochs. Passing that to resume=True does NOT
        # raise - it silently falls back to training a brand new default
        # run instead of resuming anything, so it's worth a hard stop here.
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("epoch", -1) < 0:
            raise RuntimeError(
                f"{args.resume} já é de um treino que completou todas as épocas (nada a retomar). "
                "Isso não é um checkpoint 'no meio do treino' - confira se é o arquivo certo."
            )
        print(
            f"\n*** RETOMANDO treino a partir de: {args.resume} (parou na época {ckpt['epoch'] + 1}) ***\n"
        )

        model = YOLO(str(args.resume))
        model.add_callback(
            "on_model_save", make_checkpoint_callback(args.live_copy_dir)
        )
        # resume must be the checkpoint PATH (str), not the bare bool True -
        # passing True makes Ultralytics search for "the latest run" via its
        # own heuristic instead of using this specific checkpoint.
        model.train(resume=str(args.resume))
        report_result(model)
        return

    if args.smoke_test:
        print(
            f"\n*** MODO SMOKE TEST: {args.n} imagens, {args.smoke_epochs} época(s), "
            "treino e val usam o MESMO subconjunto. ***"
        )
        print(
            "*** Métricas deste treino não significam nada - o objetivo é só confirmar que o "
            "pipeline roda e que os checkpoints .pt vão sendo gravados a cada época. ***\n"
        )
        data_yaml = build_smoke_subset(args.n)
        model_name = str(PRETRAINED_MODEL)
        train_kwargs = dict(
            data=str(data_yaml),
            epochs=args.smoke_epochs,
            imgsz=640,
            batch=min(4, args.n),
            device=device,
            project=str(RUNS_DIR),
            name="smoke_test",
            patience=0,
            plots=False,
            verbose=True,
        )
    else:
        with open(TRAIN_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not DATA_YAML.exists():
            raise RuntimeError(
                f"{DATA_YAML} não existe - rode build_dataset.py primeiro."
            )
        model_name = str(REPO_ROOT / cfg["model"])
        train_kwargs = dict(
            data=str(DATA_YAML),
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=device,
            project=str(RUNS_DIR),
            name=cfg["name"],
            patience=cfg["patience"],
            save_period=cfg.get("save_period", -1),
        )

    model = YOLO(model_name)
    model.add_callback("on_model_save", make_checkpoint_callback(args.live_copy_dir))
    model.train(**train_kwargs)
    report_result(model)


if __name__ == "__main__":
    main()
