"""
Пайплайн для модели классификации/детекции рентген-снимков.
Скачивает веса с Google Drive и выполняет инференс.
"""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Скачиваем модель
if "installer.py" not in os.listdir(HERE):
  raise EnvironmentError(
        "[!] Установщик не найден.\n"
        "    Скачайте installer.py из репозитория: "
        "https://github.com/Banan41k78/xray-contraband-detection-yolov8"
    )
  

else:
  from installer import BASE_DIR, MODEL_PATH
  
  if not os.path.exists(MODEL_PATH):
    print("[i] Веса не найдены, скачиваю...")
    import installer


import gdown
from ultralytics import YOLO
OUTPUT_DIR = Path(BASE_DIR) / "predictions"


def load_model() -> YOLO:
    """загружает модель YOLO."""
    print(f"[i] Загружаю модель: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

# Инференс 
def predict(model: YOLO, source: str,save = True):
  """
  model - YOLO модель,
  source - путь к файлу/папке/url,
  save - булевый флаг сохранять/не сохранять.
  """
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  return model.predict(source=source,
                       save = save, 
                       project=str(OUTPUT_DIR),
                       name="run",
                       exist_ok=True,
                       verbose=True,)

# Краткая сводка по предсказаниям
def summarize(results, verbose: bool = True):
    """
    Для каждой картинки:
      - при детекции  → список объектов с классом, уверенностью и bbox
      - при классификации → топ-1 класс и вероятность

    results — список объектов Results из ultralytics.
    verbose — печатать ли отчёт в stdout.
    """
    summary = []

    for r in results:
        path = str(getattr(r, "path", "?"))
        entry = {"path": path, "detections": [], "classification": None}

        # Детекция
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                xyxy = [round(float(v), 1) for v in box.xyxy[0]]
                entry["detections"].append({
                    "class": r.names[cls_id],
                    "conf": round(conf, 3),
                    "bbox": xyxy,
                })

        # Классификация
        probs = getattr(r, "probs", None)
        if probs is not None:
            entry["classification"] = {
                "class": r.names[int(probs.top1)],
                "conf": round(float(probs.top1conf), 3),
            }

        summary.append(entry)

        # Печать
        if verbose:
            print(f"\n=== {path} ===")
            if entry["detections"]:
                for d in entry["detections"]:
                    print(f"  • {d['class']:<15} conf={d['conf']:.3f}  bbox={d['bbox']}")
            elif entry["classification"]:
                c = entry["classification"]
                print(f"  Класс: {c['class']}  |  conf: {c['conf']:.3f}")
            else:
                print("  (ничего не найдено)")

    # Итоговая сводка
    if verbose and len(summary) > 1:
        total = sum(len(e["detections"]) for e in summary)
        print(f"\n[i] Обработано изображений: {len(summary)}")
        print(f"[i] Всего объектов найдено: {total}")

    return summary

    model = load_model()
    results = predict(model, sys.argv[1])
    summarize(results)
    print(f"\n[+] Результаты в: {OUTPUT_DIR.resolve()}")
