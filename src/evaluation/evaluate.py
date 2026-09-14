"""Evaluate the trained character detector on the held-out test split.

Three layers of evaluation:
  1. The standard Ultralytics `model.val(split="test")` run - official
     precision/recall/mAP50/mAP50-95, both overall AND per class (all 36
     characters). Ultralytics computes the per-class breakdown for free on
     a multi-class model - unlike the PlateDetector project, where a
     single class has nothing to break down, which is why that project's
     evaluate.py invents its own rain/time/size/legibility subgroups
     instead. Here there's no such metadata (this dataset doesn't have
     it), but the per-class table already gives a much richer picture
     than PlateDetector could ever get.
  2. Confusion pairs, read straight off Ultralytics' own confusion matrix
     (metrics.confusion_matrix.matrix) instead of reimplementing IoU
     matching from scratch - which character gets misread as which. This
     is where the classic OCR mix-ups for this alphabet (O/0, I/1, B/8,
     S/5, Q/O, Z/2...) would show up, if the model struggles with them.
  3. Plate-level exact-match rate: the actual point of this project (per
     the TCC pipeline: detect plate -> crop ROI -> read characters here)
     is reading a WHOLE plate, and one wrong character out of ~7 still
     fails the read - a metric per bounding box doesn't capture that. This
     groups each test image's boxes into reading order (row-clustered by
     y-gap, then left-to-right by x within each row - handles both
     single-row Mercosul-style plates and 2-row old-format plates without
     needing the br_old/mercosul marker that build_dataset.py already
     dropped) and compares the assembled string to ground truth.

Simplification vs. PlateDetector's evaluate.py: no --skip-subgroup /
--subgroup-only / cache-file split. That existed there because a 960x960,
4320-image test set blew up memory in a single process. This test set is
smaller (3,549 images) and already 640x640, so there's no evidence yet
it needs the same two-phase workaround - if it turns out to, add it back.

Usage:
    python src/evaluation/evaluate.py --weights outputs/runs/detect/full_run/weights/best.pt
    python src/evaluation/evaluate.py --weights outputs/runs/detect/smoke_test/weights/best.pt --limit 50
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "yolo"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = REPO_ROOT / "outputs" / "runs" / "detect"
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "eval_summary.md"

CONF_THRESHOLD = 0.25
ROW_GAP_FACTOR = (
    0.5  # y-gap (as a fraction of median char height) that starts a new row
)


def run_official_val(weights: str, batch: int):
    model = YOLO(weights)
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        project=str(RUNS_DIR),
        name="eval_test",
        batch=batch,
    )

    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    return model, metrics, overall


def per_class_table(model, metrics) -> list:
    """One row per class (all 36, even if Ultralytics' own p/r/ap arrays -
    indexed by `ap_class_index` - happen to skip a class with zero
    instances in this split): instance count, precision, recall, AP50,
    AP50-95. Sorted worst-first (by AP50-95) so problem classes surface
    at the top of the report instead of getting lost in an alphabetic table.
    """
    names = model.names
    nc = len(names)
    support = (
        {cid: int(metrics.nt_per_class[cid]) for cid in range(nc)}
        if metrics.nt_per_class is not None
        else {}
    )

    p = {cid: float("nan") for cid in range(nc)}
    r = {cid: float("nan") for cid in range(nc)}
    ap50 = {cid: float("nan") for cid in range(nc)}
    ap50_95 = {cid: float("nan") for cid in range(nc)}
    for i, cid in enumerate(metrics.box.ap_class_index):
        cid = int(cid)
        p[cid] = float(metrics.box.p[i])
        r[cid] = float(metrics.box.r[i])
        ap50[cid] = float(metrics.box.ap50[i])
        ap50_95[cid] = float(metrics.box.ap[i])

    rows = [
        {
            "name": names[cid],
            "instances": support.get(cid, 0),
            "precision": p[cid],
            "recall": r[cid],
            "ap50": ap50[cid],
            "ap50_95": ap50_95[cid],
        }
        for cid in range(nc)
    ]
    # nan != nan, so a class Ultralytics found no predictions/support for
    # sorts as -1 (worst, shown first) instead of crashing or sorting last
    rows.sort(
        key=lambda row: row["ap50_95"] if row["ap50_95"] == row["ap50_95"] else -1
    )
    return rows


def confusion_pairs(model, metrics, top_n: int = 15) -> list:
    """Top confused (true, predicted) character pairs, straight from
    Ultralytics' own confusion matrix - matrix[predicted, true] counts,
    off-diagonal only (matrix index `nc` is the "no match" row/column, for
    plain false positives/negatives, not a confusion between two real
    classes, and is excluded here)."""
    names = model.names
    cm = metrics.confusion_matrix.matrix
    nc = metrics.confusion_matrix.nc

    pairs = []
    for true_c in range(nc):
        for pred_c in range(nc):
            if true_c == pred_c:
                continue
            count = cm[pred_c, true_c]
            if count > 0:
                pairs.append((names[true_c], names[pred_c], int(count)))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_n]


def reading_order(boxes: list) -> list:
    """Order (name, xc, yc, w, h) tuples the way a person would read the
    plate: row-clustered by y (a gap bigger than ROW_GAP_FACTOR * the
    median character height in this image starts a new row), then
    left-to-right by x within each row, rows top-to-bottom. Handles both
    single-row Mercosul-style plates and 2-row old-format plates from the
    box geometry alone - build_dataset.py already dropped the
    br_old/mercosul markers that used to say which was which."""
    if not boxes:
        return []
    if len(boxes) == 1:
        return [boxes[0][0]]

    by_y = sorted(boxes, key=lambda b: b[2])
    median_h = sorted(b[4] for b in boxes)[len(boxes) // 2]
    gap = median_h * ROW_GAP_FACTOR

    rows = [[by_y[0]]]
    for b in by_y[1:]:
        if b[2] - rows[-1][-1][2] > gap:
            rows.append([b])
        else:
            rows[-1].append(b)

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[1]))
    return [b[0] for b in ordered]


def load_gt_string(label_path: Path, names: dict) -> str:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        cid, xc, yc, w, h = int(parts[0]), *map(float, parts[1:5])
        boxes.append((names[cid], xc, yc, w, h))
    return "".join(reading_order(boxes))


def run_plate_reading_eval(model, image_paths: list, conf: float, batch: int):
    names = model.names
    stems = [p.stem for p in image_paths]
    label_dir = DATASET_DIR / "labels" / "test"

    gt_strings = {
        stem: load_gt_string(label_dir / f"{stem}.txt", names) for stem in stems
    }

    # chunked predict (same reasoning as PlateDetector's evaluate.py): keeps
    # memory bounded to one chunk regardless of test set size, instead of
    # handing Ultralytics the whole image list in one predict() call.
    pred_strings = {}
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i : i + batch]
        results_chunk = model.predict(
            source=[str(p) for p in chunk], conf=conf, batch=batch, verbose=False
        )
        for result in results_chunk:
            stem = Path(result.path).stem
            boxes = [
                (names[int(cid)], xc, yc, w, h)
                for cid, (xc, yc, w, h) in zip(
                    result.boxes.cls.tolist(), result.boxes.xywhn.tolist()
                )
            ]
            pred_strings[stem] = "".join(reading_order(boxes))

    n_exact = 0
    n_len_match = 0
    n_char_correct = 0
    n_char_total = 0
    for stem in stems:
        gt = gt_strings[stem]
        pred = pred_strings.get(stem, "")
        if gt == pred:
            n_exact += 1
        if len(gt) == len(pred):
            n_len_match += 1
            n_char_total += len(gt)
            n_char_correct += sum(1 for a, b in zip(gt, pred) if a == b)

    return {
        "n_total": len(stems),
        "n_exact": n_exact,
        "n_len_match": n_len_match,
        "n_char_correct": n_char_correct,
        "n_char_total": n_char_total,
    }


def render_report(weights, test_total, overall, class_rows, pairs, plate_stats, conf):
    lines = ["# Avaliação no split de teste\n"]
    lines.append(f"- Pesos: `{weights}`")
    lines.append(f"- Imagens no split de teste: {test_total}\n")

    lines.append("## Métricas oficiais (Ultralytics `model.val(split='test')`)")
    lines.append(f"- Precision: {overall['precision']:.3f}")
    lines.append(f"- Recall: {overall['recall']:.3f}")
    lines.append(f"- mAP50: {overall['map50']:.3f}")
    lines.append(f"- mAP50-95: {overall['map50_95']:.3f}\n")

    lines.append("## Métricas por classe (pior AP50-95 primeiro)")
    lines.append("| Classe | Instâncias | Precision | Recall | AP50 | AP50-95 |")
    lines.append("|---|---|---|---|---|---|")
    for row in class_rows:
        lines.append(
            f"| {row['name']} | {row['instances']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['ap50']:.3f} | {row['ap50_95']:.3f} |"
        )
    lines.append("")

    lines.append(
        "## Pares mais confundidos (matriz de confusão oficial do Ultralytics)"
    )
    if pairs:
        lines.append("| Real | Confundido com | Ocorrências |")
        lines.append("|---|---|---|")
        for real, confused, count in pairs:
            lines.append(f"| {real} | {confused} | {count} |")
    else:
        lines.append("- Nenhuma confusão registrada.")
    lines.append("")

    lines.append(
        "## Leitura de placa completa (caracteres remontados em ordem de leitura)"
    )
    lines.append(
        f"- conf>={conf} (mesmo limiar de uso real, não a curva completa do val() oficial)"
    )
    lines.append(
        "- Ordem de leitura: agrupamento por linha (gap vertical > "
        f"{ROW_GAP_FACTOR:.0%} da altura mediana do caractere na imagem), esquerda->direita "
        "dentro da linha - cobre Mercosul (1 linha) e formato antigo (2 linhas) sem precisar "
        "da marcação br_old/mercosul (já descartada por build_dataset.py)"
    )
    n = plate_stats["n_total"]
    lines.append(f"- Placas avaliadas: {n}")
    lines.append(
        f"- Acerto exato (string completa igual): {plate_stats['n_exact']} "
        f"({plate_stats['n_exact'] / n:.1%})"
    )
    lines.append(
        f"- Nº de caracteres detectados bate com o gabarito: {plate_stats['n_len_match']} "
        f"({plate_stats['n_len_match'] / n:.1%})"
    )
    if plate_stats["n_char_total"]:
        char_acc = plate_stats["n_char_correct"] / plate_stats["n_char_total"]
        lines.append(
            f"- Acerto por caractere (só nas placas com contagem batendo): "
            f"{plate_stats['n_char_correct']}/{plate_stats['n_char_total']} ({char_acc:.1%})"
        )
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        required=True,
        help="ex: outputs/runs/detect/full_run/weights/best.pt",
    )
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="avaliar a leitura de placa completa só nas N primeiras imagens do test (debug)",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise RuntimeError(f"pesos não encontrados: {weights}")

    print(f"Pesos: {weights}")
    print("Rodando val() oficial do Ultralytics no split de teste...")
    model, metrics, overall = run_official_val(str(weights), args.batch)
    class_rows = per_class_table(model, metrics)
    pairs = confusion_pairs(model, metrics)

    all_image_paths = sorted((DATASET_DIR / "images" / "test").glob("*.jpg"))
    image_paths = all_image_paths[: args.limit] if args.limit else all_image_paths

    print(
        f"\nRemontando {len(image_paths)} placas (conf>={args.conf}) para o acerto exato..."
    )
    plate_stats = run_plate_reading_eval(model, image_paths, args.conf, args.batch)

    lines = render_report(
        weights,
        len(all_image_paths),
        overall,
        class_rows,
        pairs,
        plate_stats,
        args.conf,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRelatório salvo em {args.report}")


if __name__ == "__main__":
    main()
