"""Real-time character detection from a webcam (or video file/URL).

Opens `--source` (default 0 = default webcam) with OpenCV, runs the trained
detector frame by frame, and draws each detected character's box + class
label live. Press 'q' or Esc to quit.

Meant to point at a plate ROI (already cropped by the plate detector), not
a full scene - the training images are all close-up crops of just the
plate, so a webcam frame showing a whole car will likely detect nothing.

Usage:
    .venv/Scripts/python.exe src/inference/webcam_demo.py
    .venv/Scripts/python.exe src/inference/webcam_demo.py --mirror
    .venv/Scripts/python.exe src/inference/webcam_demo.py --source 1 --conf 0.4
    .venv/Scripts/python.exe src/inference/webcam_demo.py --source path/to/video.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from plate_reader import classify_plate, read_plate

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "character_detector.pt"

BOX_COLOR = (0, 200, 0)  # BGR
TEXT_COLOR = (255, 255, 255)


def parse_source(value):
    """--source 0/1/... -> webcam index; anything else -> file path or URL as-is."""
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--source",
        type=parse_source,
        default=0,
        help="índice da webcam (0, 1, ...) ou caminho de vídeo/URL",
    )
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="limiar de IoU do NMS (padrão do Ultralytics). Baixe (ex: 0.4) se o modelo "
        "desenhar várias caixas sobre a mesma placa.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="640 casa com o treino (configs/train.yaml)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="desfaz o efeito espelho do driver da webcam antes de detectar - "
        "sem isso, algumas webcams entregam o frame invertido horizontalmente "
        "e a placa é lida da direita pra esquerda",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise RuntimeError(
            f"pesos não encontrados: {args.weights} — rode export_model.py primeiro."
        )

    print(f"Pesos: {args.weights}")
    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"não consegui abrir a fonte de vídeo: {args.source}")

    print("Pressione 'q' ou Esc para sair.")
    prev_t = time.time()
    fps = 0.0
    last_reading = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fonte de vídeo terminou ou falhou ao ler o frame.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            predictions = model(
                frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False
            )
            result = (
                predictions[0]
                if isinstance(predictions, (list, tuple))
                else predictions
            )
            boxes = getattr(result, "boxes", None)
            detections = []
            if boxes is not None:
                for box in boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    class_name = model.names[int(box.cls[0])]
                    xc, yc, w, h = box.xywh[0].tolist()
                    detections.append((class_name, xc, yc, w, h))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BOX_COLOR, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (xmin, max(0, ymin - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        BOX_COLOR,
                        2,
                    )

            plate_text = read_plate(detections)
            plate_format = classify_plate(plate_text)
            if plate_format and plate_text != last_reading:
                print(f"Placa lida ({plate_format}): {plate_text}")
                last_reading = plate_text

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            cv2.putText(
                frame,
                f"{fps:.1f} FPS",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                2,
            )

            cv2.imshow("Detector de caracteres - YOLO11n", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' ou Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
