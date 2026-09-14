"""
Pipeline for the X-ray classification/detection model.
Downloads weights from Google Drive and runs inference.
"""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Скачиваем модель
if "installer.py" not in os.listdir(HERE):
  raise EnvironmentError(
        "[!] Installer not found.\n"
        "    Download installer.py from the repository: "
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
    """Download weights (if needed) and load the YOLO model."""
    print(f"[i] Loading model: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

# Inference 
def predict(model: YOLO, source: str,save = True):
  """
   model  — YOLO model,
   source — path to a file / folder / URL,
   save   — whether to save the annotated visualization.
  """
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  return model.predict(source=source,
                       save = save, 
                       project=str(OUTPUT_DIR),
                       name="run",
                       exist_ok=True,
                       verbose=True,)

# Prints a short prediction summary and returns a list of dicts
def summarize(results, verbose: bool = True):
    """
    For each image:
      - detection     → list of objects with class, confidence, bbox
      - classification → top-1 class and probability

    results — list of Results objects from ultralytics.
    verbose — whether to print the report to stdout.
    """
    summary = []

    for r in results:
        path = str(getattr(r, "path", "?"))
        entry = {"path": path, "detections": [], "classification": None}

        # Detection
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

        # Classification
        probs = getattr(r, "probs", None)
        if probs is not None:
            entry["classification"] = {
                "class": r.names[int(probs.top1)],
                "conf": round(float(probs.top1conf), 3),
            }

        summary.append(entry)

        # Print
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

    # Overall summary
    if verbose and len(summary) > 1:
        total = sum(len(e["detections"]) for e in summary)
        print(f"\n[i] Обработано изображений: {len(summary)}")
        print(f"[i] Всего объектов найдено: {total}")

    return summary

    model = load_model()
    results = predict(model, sys.argv[1])
    summarize(results)
    print(f"\n[+] Results saved to: {OUTPUT_DIR.resolve()}")
