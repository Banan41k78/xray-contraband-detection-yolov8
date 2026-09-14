import os
import gdown

BASE_DIR = "xray-classification"
MODEL_ID = "1UZx4Ifx-TjpYz4gplzk5Ao-vkMH3mSWM"
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

# Create the folder (won't fail if it already exists)
if not os.path.exists(BASE_DIR):
  os.makedirs(BASE_DIR, exist_ok=True)
  print(f"[+] Folder ready: {BASE_DIR}")

# Download the model if it is not there yet
if os.path.exists(MODEL_PATH):
    print(f"[i]  Model already downloaded: {MODEL_PATH}")
else:
    print("[i] Downloading model from Google Drive...")
    result = gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
    if result is None:
        raise RuntimeError("[!] Failed to download the model. Check file access.")
    print(f"[+] Model saved: {MODEL_PATH}")
    # Быстрая проверка, что файл на месте
    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"[i] File size: {size_mb:.1f} МБ")
