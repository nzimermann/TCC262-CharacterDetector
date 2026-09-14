"""Build train/val/test splits over the filtered OCR_5 annotations.

Split unit is the *source image* (`base_name`: everything before the
`.rf.<hash>` Roboflow appends to every exported filename), not the
individual exported file. The EDA (outputs/reports/eda_summary.md) found
44% of source images have copies spread across the Roboflow-provided
train/valid/test folders - sometimes with slightly different bounding
boxes (looks like re-annotation, not just augmentation) - so grouping by
`base_name` is required here, the same way build_splits.py in the
PlateDetector project groups by camera: without it, the same plate would
end up in both train and test and the test metric would mean nothing.

Only images that survive filters.py (drop br_old/mercosul, merge i->I)
enter the split pool at all - a raw file with nothing but br_old/mercosul
boxes has no character to train or evaluate on, so it's excluded
entirely, not assigned to any split.

Balancing uses the same greedy "largest group first, assign to the split
furthest below its target size" heuristic as the PlateDetector project
(a form of Longest-Processing-Time scheduling). Source images here
average ~4 copies each (nowhere near as skewed as that project's cameras,
where one alone had 3,603 images) - the heuristic is not really needed
for that reason this time, but it costs nothing and stays correct
regardless of skew, so there's no reason to write a different one.

Output: data/yolo/splits.json -> {"train": [...], "val": [...], "test": [...]}
where each entry is "<raw_split>/<filename stem>" - the ORIGINAL Roboflow
folder the file lives in under data/raw/, NOT its new split assignment.
build_dataset.py resolves this back to the actual (image, label) pair.

Usage:
    python src/data_prep/build_splits.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from filters import (
    RAW_DIR,
    SPLITS as RAW_SPLITS,
    build_id_mapping,
    filter_label_lines,
    load_class_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "yolo" / "splits.json"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "reports" / "split_summary.md"

SPLIT_NAMES = ("train", "val", "test")


def base_name(stem: str) -> str:
    """Roboflow appends '.rf.<hash>' to every exported filename; the part
    before it identifies the original source image."""
    return stem.split(".rf.")[0]


def collect_kept_items(raw_dir: Path = RAW_DIR):
    """Walk data/raw/{train,valid,test}, apply filters.py, and return:
      - groups: base_name -> list of item ids ("<raw_split>/<stem>")
      - instance_counts: item id -> number of annotations kept for it
    A raw file with no keepable annotation (only br_old/mercosul, or
    empty) never enters `groups` at all."""
    class_names = load_class_names(raw_dir)
    id_mapping = build_id_mapping(class_names)

    groups = defaultdict(list)
    instance_counts = {}
    for raw_split in RAW_SPLITS:
        split_dir = raw_dir / raw_split
        for label_path in sorted(split_dir.glob("*.txt")):
            raw_lines = label_path.read_text(encoding="utf-8").splitlines()
            kept_lines = filter_label_lines(raw_lines, id_mapping)
            if not kept_lines:
                continue
            item_id = f"{raw_split}/{label_path.stem}"
            groups[base_name(label_path.stem)].append(item_id)
            instance_counts[item_id] = len(kept_lines)
    return groups, instance_counts


def balanced_split(groups: dict, ratios=(0.70, 0.15, 0.15), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)  # break ties randomly instead of by insertion order
    items.sort(key=lambda kv: len(kv[1]), reverse=True)  # largest groups first

    total = sum(len(v) for _, v in items)
    targets = [total * r for r in ratios]
    counts = [0, 0, 0]
    assignment = {name: [] for name in SPLIT_NAMES}
    groups_per_split = {name: set() for name in SPLIT_NAMES}

    for key, item_ids in items:
        i = max(range(3), key=lambda idx: targets[idx] - counts[idx])
        split = SPLIT_NAMES[i]
        assignment[split].extend(item_ids)
        counts[i] += len(item_ids)
        groups_per_split[split].add(key)

    return assignment, groups_per_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    groups, instance_counts = collect_kept_items()
    assignment, groups_per_split = balanced_split(groups, ratios=ratios, seed=args.seed)

    # sanity check: no group (source image) split across more than one partition
    key_by_item = {
        item_id: key for key, item_ids in groups.items() for item_id in item_ids
    }
    seen = set()
    for split in SPLIT_NAMES:
        split_keys = {key_by_item[item_id] for item_id in assignment[split]}
        overlap = seen & split_keys
        assert not overlap, f"leakage detected, groups in >1 split: {overlap}"
        seen |= split_keys

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(assignment, f, indent=2)

    total_items = sum(len(v) for v in assignment.values())

    lines = ["# Split treino/val/teste (por imagem-fonte)\n"]
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Proporções alvo: {ratios}")
    lines.append(f"- Total de imagens (após o filtro de filters.py): {total_items}")
    lines.append("")
    for split in SPLIT_NAMES:
        item_ids = assignment[split]
        n_instances = sum(instance_counts[i] for i in item_ids)
        group_sizes = sorted(
            (len(groups[k]) for k in groups_per_split[split]), reverse=True
        )
        largest_share = (
            (group_sizes[0] / len(item_ids)) if item_ids and group_sizes else 0.0
        )
        lines.append(f"## {split}")
        lines.append(f"- Imagens: {len(item_ids)} ({len(item_ids) / total_items:.1%})")
        lines.append(f"- Caracteres: {n_instances}")
        lines.append(f"- Imagens-fonte distintas: {len(groups_per_split[split])}")
        lines.append(
            f"- Maior imagem-fonte do split: {group_sizes[0] if group_sizes else 0} cópias "
            f"({largest_share:.1%} do split)"
        )
        lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Split salvo em {args.output}")
    print(f"Relatório salvo em {args.report}")


if __name__ == "__main__":
    main()
