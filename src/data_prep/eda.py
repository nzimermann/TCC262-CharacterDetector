"""Exploratory data analysis over the raw OCR_5 Roboflow export (data/raw).

Reads every label .txt across train/valid/test (no sampling - each file is
tiny and images are only header-peeked, so this is cheap enough to run on
the whole dataset) and:
  - reports instance/image counts per class, splitting out uppercase `I`
    from lowercase `i` (the dataset defines them as two separate classes)
  - measures how many images carry ONLY the classes we plan to drop
    (br_old/mercosul) vs a MIX of those with real letters/digits (or `i`) -
    this decides whether "drop the whole image" is actually safe
  - confirms image dimensions (Roboflow's own export note claims a fixed
    640x640 stretch for every image)
  - measures character bbox size relative to the image (drives the imgsz
    choice for training)
  - flags source images that leak across the Roboflow-provided
    train/valid/test split (same base filename, different `.rf.<hash>`)
  - simulates the effect of the proposed filter (to be written next as
    filters.py): drop br_old/mercosul lines, merge lowercase `i` into `I`,
    drop any image left with zero annotations after that
  - saves plots to outputs/reports/figures/ and a text summary to
    outputs/reports/eda_summary.md

Usage:
    python src/data_prep/eda.py
"""

import struct
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS = ("train", "valid", "test")
FIGURES_DIR = REPO_ROOT / "outputs" / "reports" / "figures"
SUMMARY_PATH = REPO_ROOT / "outputs" / "reports" / "eda_summary.md"


def get_jpeg_size(path):
    """Read width/height straight from the JPEG SOF marker (no full decode)."""
    with open(path, "rb") as f:
        data = f.read(64 * 1024)
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            h = struct.unpack(">H", data[i + 5 : i + 7])[0]
            w = struct.unpack(">H", data[i + 7 : i + 9])[0]
            return w, h
        seglen = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + seglen
    return None


def pct(sorted_list, p):
    if not sorted_list:
        return float("nan")
    idx = min(int(len(sorted_list) * p), len(sorted_list) - 1)
    return sorted_list[idx]


def load_class_names():
    names_by_split = {}
    for split in SPLITS:
        path = RAW_DIR / split / "_darknet.labels"
        names_by_split[split] = path.read_text(encoding="utf-8").splitlines()
    first = names_by_split[SPLITS[0]]
    for split, names in names_by_split.items():
        if names != first:
            raise RuntimeError(
                f"_darknet.labels difere entre '{SPLITS[0]}' e '{split}' - "
                "os ids de classe não são comparáveis entre train/valid/test"
            )
    return first


def base_name(stem: str) -> str:
    """Roboflow appends '.rf.<hash>' to every exported filename; the part
    before it identifies the original source image - used here to detect
    the same source leaking into more than one split, or appearing more
    than once (e.g. augmented) within the same split."""
    return stem.split(".rf.")[0]


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()
    idx_I = class_names.index("I")
    idx_i = class_names.index("i")
    idx_br_old = class_names.index("br_old")
    idx_mercosul = class_names.index("mercosul")

    special_drop_ids = {idx_br_old, idx_mercosul}
    keepable_ids = set(range(len(class_names))) - special_drop_ids  # 0-35 + 'i'

    class_counts = Counter()
    images_per_split = Counter()
    anns_per_image = Counter()
    empty_label_files = 0

    only_drop_images = 0
    mixed_images = 0  # has br_old/mercosul AND has a keepable annotation too
    keepable_only_images = 0

    width_ratios, height_ratios, area_ratios = [], [], []
    image_sizes = Counter()

    base_to_splits = defaultdict(set)
    base_to_count_in_split = defaultdict(Counter)

    # simulated filter effect: merge i->I (both already 'keepable'), drop
    # br_old/mercosul lines, drop whole image if nothing survives
    kept_images = 0
    kept_instances = 0
    dropped_images_empty_after_filter = 0

    for split in SPLITS:
        split_dir = RAW_DIR / split
        label_files = sorted(split_dir.glob("*.txt"))
        for label_path in label_files:
            images_per_split[split] += 1
            base = base_name(label_path.stem)
            base_to_splits[base].add(split)
            base_to_count_in_split[base][split] += 1

            image_path = label_path.with_suffix(".jpg")
            if image_path.exists():
                dims = get_jpeg_size(image_path)
                if dims:
                    image_sizes[dims] += 1

            raw_lines = [
                line.split()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not raw_lines:
                empty_label_files += 1
            cids = [int(parts[0]) for parts in raw_lines]
            anns_per_image[len(cids)] += 1

            has_keepable = any(c in keepable_ids for c in cids)
            has_drop = any(c in special_drop_ids for c in cids)
            if has_drop and has_keepable:
                mixed_images += 1
            elif has_drop:
                only_drop_images += 1
            elif has_keepable:
                keepable_only_images += 1

            kept_here = 0
            for parts, cid in zip(raw_lines, cids):
                class_counts[cid] += 1
                if cid in special_drop_ids:
                    continue
                kept_here += 1
                kept_instances += 1
                w, h = float(parts[3]), float(parts[4])
                width_ratios.append(w)
                height_ratios.append(h)
                area_ratios.append(w * h)

            if kept_here > 0:
                kept_images += 1
            else:
                dropped_images_empty_after_filter += 1

    width_ratios.sort()
    height_ratios.sort()
    area_ratios.sort()

    total_images = sum(images_per_split.values())
    total_instances = sum(class_counts.values())

    leaked_bases = {b: s for b, s in base_to_splits.items() if len(s) > 1}
    leaked_images_count = sum(
        sum(base_to_count_in_split[b].values()) for b in leaked_bases
    )
    distinct_sources = len(base_to_splits)
    duplicated_within_split = sum(
        sum(c - 1 for c in counts.values() if c > 1)
        for counts in base_to_count_in_split.values()
    )

    # --- plots ---
    order = sorted(range(len(class_names)), key=lambda i: -class_counts[i])
    labels = [class_names[i] for i in order]
    values = [class_counts[i] for i in order]
    colors = [
        "#c44e52" if i in special_drop_ids else ("#dd8452" if i == idx_i else "#4c72b0")
        for i in order
    ]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color=colors)
    plt.title("Instâncias por classe (vermelho = descartada, laranja = mesclada em I)")
    plt.ylabel("quantidade")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(SPLITS, [images_per_split[s] for s in SPLITS], color="#dd8452")
    plt.title("Imagens por split (divisão original do Roboflow)")
    plt.ylabel("quantidade")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "images_per_split.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.hist(area_ratios, bins=60, color="#55a868")
    plt.title("Área do caractere / área da imagem (após dropar br_old/mercosul)")
    plt.xlabel("razão de área (já normalizada, formato YOLO)")
    plt.ylabel("quantidade de instâncias")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bbox_area_ratio.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    max_anns = max(anns_per_image)
    xs = list(range(max_anns + 1))
    plt.bar(xs, [anns_per_image.get(x, 0) for x in xs], color="#8172b3")
    plt.title("Caracteres anotados por imagem")
    plt.xlabel("nº de anotações na imagem")
    plt.ylabel("nº de imagens")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "chars_per_image.png", dpi=120)
    plt.close()

    # --- summary ---
    out = []
    out.append("# EDA — OCR_5 (data/raw, export YOLO Darknet)\n")
    out.append(f"- Total de imagens: **{total_images}** ({dict(images_per_split)})")
    out.append(f"- Total de instâncias anotadas: **{total_instances}**")
    out.append(f"- Arquivos de label vazios (0 anotações): {empty_label_files}\n")

    out.append("## Classes (39 no export original)")
    for i in order:
        tag = (
            " (descartada)"
            if i in special_drop_ids
            else " (mesclada em I)" if i == idx_i else ""
        )
        out.append(f"- `{class_names[i]}` (id {i}): {class_counts[i]}{tag}")
    out.append("")
    out.append(
        f"- `I` maiúsculo: {class_counts[idx_I]} / `i` minúsculo: {class_counts[idx_i]} "
        f"-> combinado (`I`+`i`): **{class_counts[idx_I] + class_counts[idx_i]}**"
    )
    out.append("")

    out.append("## Outros atributos")
    if len(image_sizes) == 1:
        (w, h), count = next(iter(image_sizes.items()))
        out.append(
            f"- Dimensão das imagens: **{w}x{h}** em 100% dos casos ({count} imagens) — pré-processamento fixo do Roboflow"
        )
    else:
        out.append(
            f"- Dimensão das imagens: {len(image_sizes)} tamanhos distintos encontrados"
        )
        for size, count in image_sizes.most_common(10):
            out.append(f"  - {size[0]}x{size[1]}: {count} imagens")
    out.append(
        f"- Imagens-fonte distintas (`base_name`, antes do `.rf.<hash>`): {distinct_sources}"
    )
    out.append(
        f"- Imagens-fonte que aparecem em mais de um split do Roboflow (vazamento train/valid/test): "
        f"{len(leaked_bases)} ({len(leaked_bases) / distinct_sources:.1%} das fontes)"
    )
    out.append(
        f"- Imagens exportadas envolvidas nesse vazamento: {leaked_images_count} "
        f"({leaked_images_count / total_images:.1%} do total)"
    )
    out.append(
        f"- Duplicatas (mesma fonte, versão diferente) dentro do MESMO split: {duplicated_within_split}"
    )
    out.append(
        "  -> build_splits.py deve agrupar por `base_name`, não reaproveitar o split "
        "train/valid/test que já vem pronto do Roboflow."
    )
    out.append("")

    out.append("## Caracteres por imagem")
    out.append(f"- Distribuição: {dict(sorted(anns_per_image.items()))}")
    out.append("")

    out.append(
        "## Tamanho da bbox em relação à imagem (todas as classes exceto br_old/mercosul)"
    )
    out.append(f"- Amostras: {len(area_ratios)}")
    out.append(
        f"- Largura relativa — mediana: {pct(width_ratios, 0.5):.4f}, "
        f"p10: {pct(width_ratios, 0.1):.4f}, p90: {pct(width_ratios, 0.9):.4f}"
    )
    out.append(
        f"- Altura relativa — mediana: {pct(height_ratios, 0.5):.4f}, "
        f"p10: {pct(height_ratios, 0.1):.4f}, p90: {pct(height_ratios, 0.9):.4f}"
    )
    out.append(
        f"- Área relativa — mediana: {pct(area_ratios, 0.5):.5f}, "
        f"p10: {pct(area_ratios, 0.1):.5f}, p90: {pct(area_ratios, 0.9):.5f}"
    )
    out.append("")

    out.append("## Efeito dos filtros propostos (ver filters.py)")
    out.append(f"- Instâncias de `br_old` removidas: {class_counts[idx_br_old]}")
    out.append(f"- Instâncias de `mercosul` removidas: {class_counts[idx_mercosul]}")
    out.append(f"- Instâncias de `i` mescladas em `I`: {class_counts[idx_i]}")
    out.append(
        f"- Imagens só com letra/dígito/`i` (nenhuma classe a descartar): {keepable_only_images}"
    )
    out.append(
        f"- Imagens só com `br_old`/`mercosul` (sem anotação válida) — descartadas inteiras: {only_drop_images}"
    )
    out.append(
        f"- Imagens mistas (`br_old`/`mercosul` JUNTO com anotação válida na mesma imagem): {mixed_images}"
        + (
            " -> confirma que descartar a imagem inteira não perde nenhuma anotação válida"
            if mixed_images == 0
            else " -> ATENÇÃO: descartar a imagem inteira perderia anotações válidas nesses casos"
        )
    )
    out.append(
        f"- Imagens restantes para treino: **{kept_images}** ({kept_images / total_images:.1%} do total)"
    )
    out.append(
        f"- Instâncias restantes para treino: **{kept_instances}** ({kept_instances / total_instances:.1%} do total)"
    )
    out.append("")

    out.append("## Figuras geradas")
    out.append("- `outputs/reports/figures/class_distribution.png`")
    out.append("- `outputs/reports/figures/images_per_split.png`")
    out.append("- `outputs/reports/figures/bbox_area_ratio.png`")
    out.append("- `outputs/reports/figures/chars_per_image.png`")

    SUMMARY_PATH.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nResumo salvo em {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
