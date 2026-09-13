# EDA — OCR_5 (data/raw, export YOLO Darknet)

- Total de imagens: **25871** ({'train': 18109, 'valid': 5176, 'test': 2586})
- Total de instâncias anotadas: **168079**
- Arquivos de label vazios (0 anotações): 0

## Classes (39 no export original)
- `O` (id 24): 8004
- `6` (id 6): 7774
- `5` (id 5): 7756
- `3` (id 3): 7553
- `4` (id 4): 7437
- `8` (id 8): 7347
- `9` (id 9): 7317
- `7` (id 7): 7176
- `2` (id 2): 6809
- `i` (id 37): 5118 (mesclada em I)
- `H` (id 17): 4864
- `P` (id 25): 4769
- `J` (id 19): 4694
- `1` (id 1): 4284
- `N` (id 23): 3975
- `M` (id 22): 3779
- `Q` (id 26): 3777
- `K` (id 20): 3766
- `A` (id 10): 3759
- `E` (id 14): 3529
- `F` (id 15): 3474
- `B` (id 11): 3335
- `U` (id 30): 3262
- `Z` (id 35): 3225
- `S` (id 28): 3171
- `C` (id 12): 3166
- `X` (id 33): 3146
- `Y` (id 34): 3124
- `G` (id 16): 3067
- `L` (id 21): 3056
- `R` (id 27): 3016
- `D` (id 13): 3002
- `T` (id 29): 2922
- `V` (id 31): 2920
- `0` (id 0): 2831
- `W` (id 32): 2737
- `I` (id 18): 2636
- `br_old` (id 36): 1380 (descartada)
- `mercosul` (id 38): 1122 (descartada)

- `I` maiúsculo: 2636 / `i` minúsculo: 5118 -> combinado (`I`+`i`): **7754**

## Outros atributos
- Dimensão das imagens: **640x640** em 100% dos casos (25871 imagens) — pré-processamento fixo do Roboflow
- Imagens-fonte distintas (`base_name`, antes do `.rf.<hash>`): 6357
- Imagens-fonte que aparecem em mais de um split do Roboflow (vazamento train/valid/test): 2810 (44.2% das fontes)
- Imagens exportadas envolvidas nesse vazamento: 20345 (78.6% do total)
- Duplicatas (mesma fonte, versão diferente) dentro do MESMO split: 15746
  -> build_splits.py deve agrupar por `base_name`, não reaproveitar o split train/valid/test que já vem pronto do Roboflow.

## Caracteres por imagem
- Distribuição: {1: 2005, 2: 167, 3: 23, 4: 16, 5: 3, 6: 39, 7: 23608, 8: 7, 12: 1, 14: 1, 20: 1}

## Tamanho da bbox em relação à imagem (todas as classes exceto br_old/mercosul)
- Amostras: 165577
- Largura relativa — mediana: 0.1250, p10: 0.1016, p90: 0.1836
- Altura relativa — mediana: 0.3516, p10: 0.1781, p90: 0.5078
- Área relativa — mediana: 0.04248, p10: 0.02051, p90: 0.07545

## Efeito dos filtros propostos (ver filters.py)
- Instâncias de `br_old` removidas: 1380
- Instâncias de `mercosul` removidas: 1122
- Instâncias de `i` mescladas em `I`: 5118
- Imagens só com letra/dígito/`i` (nenhuma classe a descartar): 23664
- Imagens só com `br_old`/`mercosul` (sem anotação válida) — descartadas inteiras: 2207
- Imagens mistas (`br_old`/`mercosul` JUNTO com anotação válida na mesma imagem): 0 -> confirma que descartar a imagem inteira não perde nenhuma anotação válida
- Imagens restantes para treino: **23664** (91.5% do total)
- Instâncias restantes para treino: **165577** (98.5% do total)

## Figuras geradas
- `outputs/reports/figures/class_distribution.png`
- `outputs/reports/figures/images_per_split.png`
- `outputs/reports/figures/bbox_area_ratio.png`
- `outputs/reports/figures/chars_per_image.png`