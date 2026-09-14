"""Materialize the filtered, split OCR_5 dataset into data/yolo/.

Reads data/yolo/splits.json (written by build_splits.py) and, for every
item in every split:
  - re-applies filters.py's line filter (drop br_old/mercosul, merge
    i->I) to the raw label file - splits.json only records which items
    were kept, not their filtered content, so this is where that content
    actually gets computed and written
  - hardlinks (falls back to copy) the source image into
    data/yolo/images/<split>/
  - writes the filtered/remapped label file into data/yolo/labels/<split>/

Unlike PlateDetector's convert_annotations.py, there is no bbox math here:
the raw annotations are already YOLO-normalized `<class> xc yc w h`, so
this script only filters and relocates files - it doesn't compute
anything geometric.

Also writes data/yolo/data.yaml, ready for `YOLO(...).train(data=...)`.

Usage:
    python src/data_prep/build_dataset.py
"""

import json
import os
import shutil
from pathlib import Path

from filters import (
    RAW_DIR,
    build_id_mapping,
    filter_label_lines,
    final_class_names,
    load_class_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_PATH = REPO_ROOT / "data" / "yolo" / "splits.json"
OUT_DIR = REPO_ROOT / "data" / "yolo"


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink: instant, no extra disk space on the same volume
    except OSError:
        shutil.copy2(src, dst)


def convert_split(split_name: str, item_ids: list, raw_dir: Path, id_mapping: dict):
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_ok, n_skipped, n_chars = 0, 0, 0
    for i, item_id in enumerate(item_ids, 1):
        raw_split, stem = item_id.split("/", 1)
        img_src = raw_dir / raw_split / f"{stem}.jpg"
        lbl_src = raw_dir / raw_split / f"{stem}.txt"

        if not img_src.exists() or not lbl_src.exists():
            print(f"[aviso] arquivo ausente, pulando: {item_id}")
            n_skipped += 1
            continue

        raw_lines = lbl_src.read_text(encoding="utf-8").splitlines()
        kept_lines = filter_label_lines(raw_lines, id_mapping)
        if not kept_lines:
            # shouldn't happen - build_splits.py already filtered this item -
            # but skip defensively instead of writing an empty label file
            print(f"[aviso] sem anotação válida após filtro, pulando: {item_id}")
            n_skipped += 1
            continue

        link_or_copy(img_src, img_out / f"{stem}.jpg")
        (lbl_out / f"{stem}.txt").write_text(
            "\n".join(kept_lines) + "\n", encoding="utf-8"
        )

        n_ok += 1
        n_chars += len(kept_lines)
        if i % 5000 == 0:
            print(f"  [{split_name}] {i}/{len(item_ids)} imagens processadas...")

    print(
        f"[{split_name}] OK: {n_ok} imagens, {n_chars} caracteres, {n_skipped} puladas"
    )
    return n_ok, n_chars


def write_data_yaml(class_names_final: list):
    yaml_path = OUT_DIR / "data.yaml"
    # every name here must be quoted: several are bare digits ('0'..'9'),
    # which YAML would otherwise parse as integers instead of class names.
    names_str = ", ".join(f"'{name}'" for name in class_names_final)
    content = (
        f"path: {OUT_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names_final)}\n"
        f"names: [{names_str}]\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"data.yaml salvo em {yaml_path}")


def main():
    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)

    class_names = load_class_names(RAW_DIR)
    id_mapping = build_id_mapping(class_names)
    final_names = final_class_names(class_names, id_mapping)

    summary = {}
    for split_name in ("train", "val", "test"):
        summary[split_name] = convert_split(
            split_name, splits[split_name], RAW_DIR, id_mapping
        )

    write_data_yaml(final_names)

    print("\nResumo:")
    for split_name, (n_ok, n_chars) in summary.items():
        print(f"  {split_name}: {n_ok} imagens, {n_chars} caracteres")


if __name__ == "__main__":
    main()
