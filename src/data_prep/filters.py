"""Filtering rules applied before converting annotations to YOLO format.

Rules (numbers behind these came from outputs/reports/eda_summary.md):
  - drop any `br_old` or `mercosul` annotation line entirely - they mark
    plate type, not a character, and are not part of the 36 classes this
    model detects.
  - merge lowercase `i` into uppercase `I` - the dataset's own export
    treats them as two different classes, but it's the same letter for the
    purpose of reading a plate. The EDA found 5,118 `i` vs 2,636 `I`
    instances; merged they land in the normal range for a letter class.
  - drop the whole image only if nothing is left after that. The EDA
    confirmed 0 images mix `br_old`/`mercosul` with a real letter/digit -
    every image with one of these two classes has ONLY that class, so
    dropping the image is equivalent to dropping the (otherwise pointless)
    line; no real annotation is ever lost by this rule.

Final class ids after filtering land on exactly 0-35 (10 digits + 26
letters, `i` folded into `I`) with the SAME ids they already had in the
raw export (nothing before class 36 gets dropped or shifted) - this is a
property of this specific dataset's class ordering, not something this
module assumes going in; build_id_mapping() derives it from
_darknet.labels rather than hardcoding ids.
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS = ("train", "valid", "test")

DROP_CLASSES = {"br_old", "mercosul"}
MERGE_CLASSES = {"i": "I"}


def load_class_names(raw_dir: Path = RAW_DIR, splits=SPLITS) -> list:
    """Read _darknet.labels and return the 39 class names in id order.
    Asserts the file is identical across train/valid/test, since ids are
    only meaningful if every split agrees on what they mean."""
    names_by_split = {}
    for split in splits:
        path = raw_dir / split / "_darknet.labels"
        names_by_split[split] = path.read_text(encoding="utf-8").splitlines()
    first = names_by_split[splits[0]]
    for split, names in names_by_split.items():
        if names != first:
            raise RuntimeError(
                f"_darknet.labels difere entre '{splits[0]}' e '{split}' - "
                "os ids de classe não são comparáveis entre train/valid/test"
            )
    return first


def build_id_mapping(class_names: list) -> dict:
    """Map old class id (0-38, as exported by Roboflow) -> new class id
    (0-35, final training classes). An old id that should be DROPPED
    (br_old/mercosul) is simply absent from the returned dict."""
    name_to_new_id = {}
    next_id = 0
    for name in class_names:
        target = MERGE_CLASSES.get(name, name)
        if target in DROP_CLASSES:
            continue
        if target not in name_to_new_id:
            name_to_new_id[target] = next_id
            next_id += 1

    mapping = {}
    for old_id, name in enumerate(class_names):
        if name in DROP_CLASSES:
            continue
        target = MERGE_CLASSES.get(name, name)
        mapping[old_id] = name_to_new_id[target]
    return mapping


def final_class_names(class_names: list, id_mapping: dict) -> list:
    """The 36 final class names, in new-id order, using the canonical
    (non-merged) name for each id - e.g. new id 18 is reported as `I`,
    never `i`, even though both old ids map to it."""
    names = [None] * len(set(id_mapping.values()))
    for old_id, new_id in id_mapping.items():
        name = class_names[old_id]
        if name in MERGE_CLASSES:  # alias, not the canonical name for this id
            continue
        names[new_id] = name
    return names


def filter_label_lines(lines, id_mapping: dict) -> list:
    """Apply id_mapping to raw YOLO label lines: drop lines whose class
    isn't in the mapping (br_old/mercosul), remap the rest. An empty
    result means the image should be dropped entirely."""
    kept = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        old_id = int(parts[0])
        if old_id not in id_mapping:
            continue
        new_id = id_mapping[old_id]
        kept.append(" ".join([str(new_id), *parts[1:]]))
    return kept


def keep_image(lines, id_mapping: dict) -> bool:
    """Whether an image (its raw label lines) survives filtering at all."""
    return len(filter_label_lines(lines, id_mapping)) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    class_names = load_class_names(args.raw_dir)
    id_mapping = build_id_mapping(class_names)

    print("Mapeamento de classes (id antigo -> novo):")
    for old_id, name in enumerate(class_names):
        new_id = id_mapping.get(old_id)
        status = f"-> {new_id}" if new_id is not None else "(descartada)"
        print(f"  {old_id:2d} {name:8s} {status}")

    print(f"\nClasses finais ({len(final_class_names(class_names, id_mapping))}):")
    print(final_class_names(class_names, id_mapping))

    total_images = 0
    kept_images = 0
    total_lines = 0
    kept_lines = 0
    for split in SPLITS:
        split_dir = args.raw_dir / split
        for label_path in sorted(split_dir.glob("*.txt")):
            total_images += 1
            raw_lines = label_path.read_text(encoding="utf-8").splitlines()
            total_lines += len([line for line in raw_lines if line.strip()])
            new_lines = filter_label_lines(raw_lines, id_mapping)
            kept_lines += len(new_lines)
            if new_lines:
                kept_images += 1

    print(
        f"\nImagens: {kept_images}/{total_images} restantes ({kept_images / total_images:.1%})"
    )
    print(
        f"Instâncias: {kept_lines}/{total_lines} restantes ({kept_lines / total_lines:.1%})"
    )
