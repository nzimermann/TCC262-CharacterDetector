# Avaliação no split de teste

- Pesos: `models\character_detector.pt`
- Imagens no split de teste: 3549

## Métricas oficiais (Ultralytics `model.val(split='test')`)
- Precision: 0.990
- Recall: 0.989
- mAP50: 0.992
- mAP50-95: 0.774

## Métricas por classe (pior AP50-95 primeiro)
| Classe | Instâncias | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| I | 1245 | 0.989 | 0.984 | 0.990 | 0.660 |
| J | 737 | 0.966 | 0.989 | 0.988 | 0.732 |
| Q | 547 | 0.959 | 0.967 | 0.979 | 0.738 |
| P | 861 | 0.994 | 0.998 | 0.995 | 0.742 |
| 7 | 1269 | 1.000 | 0.990 | 0.995 | 0.746 |
| 1 | 562 | 0.986 | 0.989 | 0.982 | 0.753 |
| 5 | 1278 | 0.995 | 0.991 | 0.993 | 0.755 |
| F | 512 | 1.000 | 1.000 | 0.995 | 0.756 |
| A | 481 | 1.000 | 0.994 | 0.995 | 0.761 |
| M | 645 | 0.972 | 0.987 | 0.979 | 0.763 |
| 0 | 448 | 0.971 | 0.987 | 0.991 | 0.764 |
| E | 439 | 0.999 | 0.991 | 0.995 | 0.764 |
| 4 | 1356 | 0.996 | 0.998 | 0.994 | 0.767 |
| H | 926 | 0.999 | 0.992 | 0.994 | 0.768 |
| L | 402 | 0.982 | 0.942 | 0.989 | 0.769 |
| 6 | 1216 | 0.998 | 0.992 | 0.995 | 0.769 |
| 9 | 1228 | 0.995 | 0.996 | 0.994 | 0.770 |
| Y | 336 | 0.999 | 1.000 | 0.995 | 0.778 |
| 8 | 1234 | 0.993 | 0.989 | 0.995 | 0.778 |
| 2 | 1062 | 0.994 | 0.990 | 0.991 | 0.779 |
| O | 1491 | 0.986 | 0.970 | 0.991 | 0.779 |
| K | 551 | 0.999 | 1.000 | 0.995 | 0.781 |
| 3 | 1171 | 0.991 | 0.987 | 0.994 | 0.781 |
| N | 595 | 0.983 | 0.993 | 0.988 | 0.781 |
| C | 449 | 0.985 | 0.991 | 0.995 | 0.784 |
| V | 318 | 0.994 | 0.996 | 0.991 | 0.784 |
| Z | 359 | 0.996 | 1.000 | 0.994 | 0.785 |
| G | 341 | 0.996 | 0.994 | 0.994 | 0.795 |
| B | 464 | 0.975 | 0.989 | 0.986 | 0.800 |
| S | 344 | 0.992 | 1.000 | 0.993 | 0.802 |
| T | 276 | 1.000 | 0.997 | 0.995 | 0.802 |
| D | 344 | 0.968 | 0.974 | 0.993 | 0.804 |
| R | 300 | 0.992 | 0.977 | 0.995 | 0.808 |
| X | 375 | 0.999 | 1.000 | 0.995 | 0.816 |
| W | 319 | 0.999 | 0.984 | 0.993 | 0.822 |
| U | 350 | 0.999 | 0.994 | 0.995 | 0.832 |

## Pares mais confundidos (matriz de confusão oficial do Ultralytics)
| Real | Confundido com | Ocorrências |
|---|---|---|
| L | J | 27 |
| O | Q | 23 |
| Q | O | 20 |
| 8 | B | 15 |
| J | L | 11 |
| M | N | 10 |
| O | 0 | 10 |
| O | D | 9 |
| H | M | 8 |
| 6 | 5 | 7 |
| 0 | O | 6 |
| 1 | 2 | 6 |
| 3 | C | 6 |
| O | 8 | 6 |
| B | 8 | 5 |

## Leitura de placa completa (caracteres remontados em ordem de leitura)
- conf>=0.25 (mesmo limiar de uso real, não a curva completa do val() oficial)
- Ordem de leitura: agrupamento por linha (gap vertical > 50% da altura mediana do caractere na imagem), esquerda->direita dentro da linha - cobre Mercosul (1 linha) e formato antigo (2 linhas) sem precisar da marcação br_old/mercosul (já descartada por build_dataset.py)
- Placas avaliadas: 3549
- Acerto exato (string completa igual): 3164 (89.2%)
- Nº de caracteres detectados bate com o gabarito: 3278 (92.4%)
- Acerto por caractere (só nas placas com contagem batendo): 22772/22946 (99.2%)
